import asyncio

from app.approvals import ApprovalBroker, Decision
from app.tools.registry import Risk


async def _pending(broker, risk=Risk.READ, confirm=None):
    task = asyncio.create_task(broker.wait("a1", risk, confirm))
    await asyncio.sleep(0)
    return task


async def test_approve():
    broker = ApprovalBroker(timeout_s=1)
    task = await _pending(broker)
    assert broker.resolve("a1", "approve") is None
    assert await task is Decision.APPROVE


async def test_expires_when_no_answer():
    broker = ApprovalBroker(timeout_s=0.05)
    assert await broker.wait("a1", Risk.READ, None) is Decision.EXPIRED
    assert broker.resolve("a1", "approve") is not None  # too late


async def test_always_allow_only_for_reads():
    broker = ApprovalBroker(timeout_s=1)
    task = await _pending(broker, Risk.CHANGE)
    assert "only available" in broker.resolve("a1", "always_allow")
    assert not task.done()
    broker.resolve("a1", "reject")
    assert await task is Decision.REJECT


async def test_destructive_needs_typed_confirmation():
    broker = ApprovalBroker(timeout_s=1)
    task = await _pending(broker, Risk.DESTRUCTIVE, confirm="db")
    assert broker.resolve("a1", "approve") is not None
    assert broker.resolve("a1", "approve", "dbx") is not None
    assert not task.done()
    assert broker.resolve("a1", "approve", "db") is None
    assert await task is Decision.APPROVE


async def test_unknown_id_and_decision():
    broker = ApprovalBroker(timeout_s=1)
    assert broker.resolve("nope", "approve") is not None
    task = await _pending(broker)
    assert broker.resolve("a1", "expired") is not None
    assert broker.resolve("a1", "yolo") is not None
    broker.reject_all()
    assert await task is Decision.REJECT


def test_registry_supports(registry):
    assert registry.supports("docker", "read")
    assert registry.supports("docker", "change")
    assert not registry.supports("kubernetes", "read")


async def test_session_allow_only_when_the_action_offered_it():
    broker = ApprovalBroker(timeout_s=1)
    task = await _pending(broker, Risk.CHANGE)  # a change that didn't offer it (e.g. restart)
    assert "isn't available" in broker.resolve("a1", "session_allow")  # a modified client can't force it
    assert not task.done()
    broker.resolve("a1", "reject")
    await task

    offered = asyncio.create_task(broker.wait("a1", Risk.CHANGE, None, session_allowable=True))
    await asyncio.sleep(0)
    assert broker.resolve("a1", "session_allow") is None
    assert await offered is Decision.SESSION_ALLOW
