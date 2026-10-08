from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from typing_extensions import TypeVar

from pydantic_ai._run_context import CapabilityEventT, CustomEventT
from pydantic_ai.capabilities.abstract import select_workspace
from pydantic_ai.durable_exec import SerializedRunContext
from pydantic_ai.durable_exec._toolset import EnqueueGuard, enqueue_not_supported_message
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent, CustomEvent
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import Workspace, WorkspaceRef
from pydantic_ai.workspaces.unavailable import NO_WORKSPACE, UnavailableWorkspace

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

AgentDepsT = TypeVar('AgentDepsT', default=object, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""


# Payloads written by a worker running an older version, or by a custom `serialize_run_context` that
# still spells the old name. An activity can be dispatched by one worker version and replayed by
# another, so the old key has to keep arriving at the renamed field — otherwise the field lands in
# `__dict__` under a name nothing reads, and `capability_active` raises as "not carried".
_RENAMED_FIELDS: tuple[tuple[str, str], ...] = (
    ('capability_loaded', 'capability_active'),
    ('available_capability_ids', 'active_capability_ids'),
)


class TemporalRunContext(SerializedRunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

    By default, only the `deps`, `run_id`, `conversation_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `usage`, `usage_limits`, `partial_output`, `trace_include_content`, `instrumentation_version`, `loaded_capability_ids`, `discovered_tool_names`, the private dispatch-only availability supplements, and `capability_active` attributes will be available. Reading any other attribute raises a `UserError` explaining how to make it available, rather than returning its default value, so a field that didn't cross the boundary can't be mistaken for real run state.

    `agent` and `root_capability` are re-attached from the worker's agent instance, `pending_messages` holds a guard that makes [`enqueue`][pydantic_ai.tools.RunContext.enqueue] raise inside an activity, and `tool_manager` and `realtime_session` are `None`: they hold live run state that isn't serializable (for `tool_manager`, `available_tool_names` returns the resolved snapshot serialized at activity dispatch time, falling back to `discovered_tool_names` if a custom subclass doesn't carry it; for `realtime_session`, `None` already means "not available here"). The `capabilities` registry is excluded for the same reason — it holds live capability objects (toolsets, hooks, callables) — so `active_capability_ids` likewise returns a snapshot serialized at dispatch time, which is what lets [`is_tool_available`][pydantic_ai.tools.RunContext.is_tool_available] answer for a capability-owned tool inside an activity; reading `capabilities` itself still raises. `model` and `tracer` are excluded as live objects too. `messages` is excluded because the full history would be duplicated into every activity payload, and `prompt` is excluded because a multi-modal prompt can carry large `BinaryContent` that would likewise ride in every activity payload, risking Temporal's 2 MB limit. `model_settings` is excluded because it's only set for model requests, which receive it as their own activity parameter, and `validation_context` because it's an arbitrary user object with no serialization contract. A live `workspace` cannot cross the activity boundary either: only its [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] is serialized, and the activity rebuilds `workspace` from it through the agent's capabilities (their `get_workspace`, which may read only `deps` and the fields listed here), policy wrappers included. A subclass whose `deserialize_run_context` sets `workspace` itself keeps that value.
    To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]. A subclass can use this escape hatch to opt in to carrying `prompt` if it knows its prompts are text-only.
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        for old_name, new_name in _RENAMED_FIELDS:
            # Keyed on presence, not truthiness: `capability_active` is `None` for every activity
            # dispatched outside capability dispatch — the common case — and a value-based guard
            # would drop the key there, leaving the renamed field absent and the guard below
            # reporting it as one that never crossed the boundary.
            if old_name in kwargs:
                kwargs.setdefault(new_name, kwargs.pop(old_name))
        super().__init__(deps, **kwargs)

    @classmethod
    def _missing_field_message(cls, name: str) -> str:
        return (
            f'{name!r} is not available on {cls.__name__!r} inside a Temporal activity. '
            'To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to `TemporalDurability`.'
        )

    def _expose_field(self, name: str) -> None:
        """Mark a framework-attached field as readable after deserialization."""
        instance_fields = object.__getattribute__(self, '__dataclass_fields__')
        instance_fields[name] = RunContext.__dataclass_fields__[name]

    @overload
    async def emit(self, event: CustomEventT, /) -> CustomEventT: ...

    @overload
    async def emit(self, event: CapabilityEventT, /) -> CapabilityEventT: ...

    async def emit(self, event: CustomEvent | CapabilityEvent, /) -> CustomEvent | CapabilityEvent:
        """Reject `emit` from inside a Temporal activity.

        Tools and event stream handlers run inside activities where the run's event stream isn't
        reachable, so events emitted there can't currently flow back into the stream; raising beats
        silently dropping them. This covers a capability's own tools emitting a `CapabilityEvent`,
        which run in activities like any other tool. Events emitted workflow-side (e.g. from
        capability hooks) work as usual.

        Lifting this needs a transport from the activity back to the workflow and a decision on what
        a retried attempt's events mean; tracked in https://github.com/pydantic/pydantic-ai/issues/7971.
        """
        raise UserError(
            'Emitting events from a tool or event stream handler is not supported under Temporal yet, as '
            'they run inside activities that cannot reach the run event stream. This includes a capability '
            'emitting a `CapabilityEvent` from one of its own tools. Emit events from capability hooks, '
            'which run in the workflow, instead. Tracked in '
            'https://github.com/pydantic/pydantic-ai/issues/7971.'
        )

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context to a `dict[str, Any]`."""
        serialized = super().serialize_run_context(ctx)
        # Only the reference crosses into the activity; the live handle stays in the workflow, and
        # `TemporalRunContext.__init__` supplies the inert default when no reference was serialized.
        if (workspace_ref := ctx.workspace.ref) is not None:
            serialized['workspace_ref'] = workspace_ref
        return serialized

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> TemporalRunContext[Any]:
        """Deserialize the run context from a `dict[str, Any]`."""
        return cls(**ctx, deps=deps)


def deserialize_run_context(
    run_context_type: type[TemporalRunContext[Any]],
    serialized: dict[str, Any],
    *,
    deps: Any,
    agent: AbstractAgent[Any, Any] | None,
) -> RunContext[Any]:
    """Deserialize a run context and attach the agent instance.

    This is a helper used internally by the Temporal wrappers. It calls the
    (potentially user-overridden) `TemporalRunContext.deserialize_run_context`
    and then sets `agent` and `root_capability` on the result so custom subclasses
    don't need to know about either parameter. Setting `root_capability` lets the
    durability capability fire the capability chain against the live model stream
    inside the activity, which is required for capabilities like
    `ProcessEventStream` to see real (non-replayed) events.
    """
    ctx = run_context_type.deserialize_run_context(serialized, deps=deps)
    if agent is not None:
        ctx.__dict__['agent'] = agent
        ctx.__dict__['root_capability'] = agent.root_capability
        _restore_workspace(ctx, agent)
    # `pending_messages` isn't serialized across the activity boundary, and any code running inside
    # an activity (a tool, a `process_tool_call` hook, an `event_stream_handler`) is in a durable
    # unit whose result is replayed without re-running it, so an enqueue would be dropped. Install
    # the same guard the in-process engines use so `ctx.enqueue()` raises the shared explanation.
    ctx.__dict__['pending_messages'] = EnqueueGuard(enqueue_not_supported_message('activity', 'workflow'))
    return ctx


def _restore_workspace(ctx: RunContext[Any], agent: AbstractAgent[Any, Any]) -> None:
    """Rebuild `ctx.workspace` from the serialized `WorkspaceRef` through the worker's capabilities.

    The same selection the workflow made, minus the durable wrapper: an activity is the durable
    unit, so its calls reach the backend directly. Policy the capabilities apply (a
    `ReadOnlyWorkspace`, a user wrapper) comes back with it, which is why the ref alone crosses. A
    workspace the subclass already restored is left alone, and a payload without a ref keeps the
    placeholder: the run has no workspace, or a fresh one whose `ensure` activity is the one
    building it.
    """
    workspace = ctx.__dict__.get('workspace')
    ref = ctx.__dict__.get('workspace_ref')
    if not isinstance(workspace, Workspace) or workspace.backend is not NO_WORKSPACE:
        return
    if not isinstance(ref, WorkspaceRef):
        return
    restored = select_workspace(agent.root_capability, ctx, ref=ref)
    if restored is None:
        restored = Workspace(
            UnavailableWorkspace(
                f'No capability on agent {agent.name!r} can supply workspace {ref.id!r} from provider '
                f'{ref.provider!r} inside this Temporal activity: every `get_workspace` returned `None`. The '
                'worker must be constructed with the same workspace capabilities as the workflow.'
            )
        )
    ctx.__dict__['workspace'] = restored
