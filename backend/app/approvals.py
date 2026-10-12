"""Approval gate. A proposed action waits here until the user decides; nothing runs before that."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from app.tools.registry import Risk


class Decision(str, Enum):
    APPROVE = "approve"
    ALWAYS_ALLOW = "always_allow"
    # Run this, and don't ask again in this chat session for the same group of tools on the same
    # project. Only accepted for actions that offered it (checked here, not just hidden in the UI).
    SESSION_ALLOW = "session_allow"
    REJECT = "reject"
    EXPIRED = "expired"


@dataclass
class _Pending:
    risk: Risk
    confirm_text: str | None
    future: asyncio.Future[Decision]
    session_allowable: bool = False


class ApprovalBroker:
    """Holds one connection's pending actions.

    Approval is bound to an action id; the agent executes the arguments it stored for
    that id, so the client can only say yes or no, never change what runs.
    """

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self._pending: dict[str, _Pending] = {}

    async def wait(self, action_id: str, risk: Risk, confirm_text: str | None,
                   session_allowable: bool = False) -> Decision:
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        self._pending[action_id] = _Pending(risk, confirm_text, future, session_allowable)
        try:
            return await asyncio.wait_for(future, self.timeout_s)
        except TimeoutError:
            return Decision.EXPIRED
        finally:
            self._pending.pop(action_id, None)

    def resolve(self, action_id: str, decision: str, confirm_text: str | None = None) -> str | None:
        """Record the user's decision. Returns an error message if it was not accepted."""
        pending = self._pending.get(action_id)
        if pending is None or pending.future.done():
            return "This action is no longer pending (it may have expired)."
        try:
            choice = Decision(decision)
        except ValueError:
            return f"Unknown decision {decision!r}."
        if choice is Decision.EXPIRED:
            return f"Unknown decision {decision!r}."
        if choice is Decision.ALWAYS_ALLOW and pending.risk is not Risk.READ:
            return "'Always allow' is only available for read-only actions."
        if choice is Decision.SESSION_ALLOW and not pending.session_allowable:
            return "'Allow for this session' isn't available for this action."
        if (
            choice is not Decision.REJECT
            and pending.confirm_text is not None
            and (confirm_text or "").strip() != pending.confirm_text
        ):
            return f"Type '{pending.confirm_text}' exactly to confirm."
        pending.future.set_result(choice)
        return None

    def reject_all(self) -> None:
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_result(Decision.REJECT)
