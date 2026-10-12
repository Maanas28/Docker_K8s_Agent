"""Usage counter for the free cloud models, so a model is never used past its free limit.

Every HTTP request to a cloud provider goes through a Meter (see llm_openai.py):

1. acquire(): before sending, check the pool's limits. Per-minute limits are waited out (with a
   countdown in the chat); a daily limit that is down to the reserve (2 calls by default) stops the
   request with QuotaExhausted. Passing the check reserves the request, so requests in flight are
   counted too and two chats can't both take the last calls.
2. release(): after the provider answered, the reservation becomes real usage, with the tokens the
   provider reported (estimated only when it reports none). A request the provider refused with a
   429, or one that never reached it, isn't counted. A daily 429 marks the pool used up until its
   reset. Where the provider reports its own "remaining" figure (Groq), that figure wins.

Daily counts live in SQLite (they survive restarts) under the provider's own day, which resets at
midnight in the pool's time zone. Per-minute windows are kept in memory (a restart forgets at most
the last 60 seconds). One QuotaTracker is shared by every chat in the process.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Mapping
from zoneinfo import ZoneInfo

from app.config import PoolSpec
from app.store import Store

DEFAULT_TOKENS_PER_CALL = 4000  # until a pool has its own average (measured: 4-8k per agent call)
MIN_CALLS_FOR_AVERAGE = 3


class QuotaExhausted(Exception):
    """The model can't be used now (daily limit at the reserve, or a call bigger than a limit)."""


class StopWaiting(Exception):
    """The user pressed Stop while the agent waited for a per-minute limit."""


@dataclass
class Reservation:
    pool: str
    day: str
    est_tokens: int
    entry: list[float] = field(default_factory=list)  # [time, tokens] in the minute window


@dataclass
class HeaderSpec:
    """Which response headers carry a provider's own daily request count (None = it sends none)."""

    limit_requests_day: str | None = None
    remaining_requests_day: str | None = None


class QuotaTracker:
    def __init__(self, store: Store, pools: Mapping[str, PoolSpec], reserve_calls: int,
                 clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.pools = dict(pools)
        self.reserve_calls = reserve_calls
        self.clock = clock
        self._inflight: dict[str, list[Reservation]] = defaultdict(list)
        self._minute: dict[str, deque[list[float]]] = defaultdict(deque)
        # A limit the provider reported in its headers, when it differs from agent.toml.
        self._limit_override: dict[str, int] = {}

    # ---- time ---------------------------------------------------------------------------------

    def _zone(self, pool: str) -> ZoneInfo:
        return ZoneInfo(self.pools[pool].reset_timezone)

    def day(self, pool: str) -> str:
        return datetime.fromtimestamp(self.clock(), self._zone(pool)).date().isoformat()

    def resets_at(self, pool: str) -> datetime:
        now = datetime.fromtimestamp(self.clock(), self._zone(pool))
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def _reset_label(self, pool: str) -> str:
        at = self.resets_at(pool).astimezone()  # shown in the user's own time zone
        today = datetime.fromtimestamp(self.clock()).astimezone().date()
        return at.strftime("%H:%M") if at.date() == today else at.strftime("%H:%M tomorrow")

    # ---- the numbers ----------------------------------------------------------------------------

    def snapshot(self, pool: str, est_tokens: int | None = None) -> dict[str, Any]:
        """Everything the dropdown and the checks need, counting requests in flight as used."""
        spec = self.pools[pool]
        row = self.store.usage(pool, self.day(pool))
        inflight = [r for r in self._inflight[pool] if r.day == row.day]
        used_requests = row.requests + len(inflight)
        used_tokens = row.tokens + sum(r.est_tokens for r in inflight)
        rpd = self._limit_override.get(pool, spec.requests_per_day)
        average = (
            row.tokens / row.requests if row.requests >= MIN_CALLS_FOR_AVERAGE
            else (est_tokens or DEFAULT_TOKENS_PER_CALL)
        )
        requests_left = max(0, rpd - used_requests) if rpd else None
        tokens_left = max(0, spec.tokens_per_day - used_tokens) if spec.tokens_per_day else None
        candidates = [n for n in (
            requests_left,
            int(tokens_left // max(average, 1)) if tokens_left is not None else None,
        ) if n is not None]
        calls_left = min(candidates) if candidates else None
        return {
            "day": row.day,
            "requests_used": used_requests,
            "requests_limit": rpd,
            "requests_left": requests_left,
            "tokens_used": used_tokens,
            "tokens_limit": spec.tokens_per_day,
            "tokens_left": tokens_left,
            "tokens_estimated": row.estimated_tokens,
            "average_tokens_per_call": round(average),
            "calls_left": 0 if row.blocked else calls_left,
            "blocked": row.blocked,
            "synced_at": row.synced_at.isoformat() if row.synced_at else None,
            "resets_at": self.resets_at(pool).isoformat(),
            "resets_label": self._reset_label(pool),
            "requests_per_minute": spec.requests_per_minute,
            "tokens_per_minute": spec.tokens_per_minute,
        }

    def usable(self, pool: str) -> tuple[bool, str]:
        """(can it be chosen/used now, why not). Down to the reserve counts as not usable."""
        snap = self.snapshot(pool)
        if snap["blocked"]:
            return False, f"used up today · resets {snap['resets_label']}"
        left = snap["calls_left"]
        if left is not None and left <= self.reserve_calls:
            return False, f"{left} calls left, kept in reserve · resets {snap['resets_label']}"
        return True, ""

    # ---- before and after each request ----------------------------------------------------------

    def check(self, pool: str, label: str, est_tokens: int) -> float:
        """Seconds to wait for a per-minute limit (0 = send now). Raises QuotaExhausted."""
        spec = self.pools[pool]
        snap = self.snapshot(pool, est_tokens)
        if snap["blocked"]:
            raise QuotaExhausted(
                f"{label} has used up its free requests for today (the provider refused more). "
                f"It resets at {snap['resets_label']}. Pick another model in the dropdown and say "
                "'continue'."
            )
        left = snap["calls_left"]
        if left is not None and left <= self.reserve_calls:
            raise QuotaExhausted(
                f"{label} is down to about {left} free calls for today, so I stopped here before it "
                f"runs out (nothing was left half-done). It resets at {snap['resets_label']}. Pick "
                "another model in the dropdown and say 'continue'."
            )
        if spec.tokens_per_minute and est_tokens > spec.tokens_per_minute:
            raise QuotaExhausted(
                f"Each call is now about {est_tokens:,} tokens, more than {label}'s limit of "
                f"{spec.tokens_per_minute:,} tokens per minute, so it can't be sent. Start a new chat, "
                "or pick a model with a higher limit."
            )
        return self._minute_wait(pool, est_tokens)

    def _minute_wait(self, pool: str, est_tokens: int) -> float:
        spec = self.pools[pool]
        now = self.clock()
        window = self._minute[pool]
        while window and window[0][0] <= now - 60:
            window.popleft()
        waits = [0.0]
        if spec.requests_per_minute and len(window) >= spec.requests_per_minute:
            waits.append(window[len(window) - spec.requests_per_minute][0] + 60 - now)
        if spec.tokens_per_minute:
            excess = sum(e[1] for e in window) + est_tokens - spec.tokens_per_minute
            for at, tokens in window:  # oldest first: wait until enough tokens leave the window
                if excess <= 0:
                    break
                excess -= tokens
                waits.append(at + 60 - now)
        return max(waits)

    def reserve(self, pool: str, est_tokens: int) -> Reservation:
        """Call right after check() returned 0 (no await in between, so nothing can slip in)."""
        res = Reservation(pool, self.day(pool), est_tokens, [self.clock(), float(est_tokens)])
        self._inflight[pool].append(res)
        self._minute[pool].append(res.entry)
        return res

    def release(self, res: Reservation, *, counted: bool, tokens: int | None = None,
                headers: Mapping[str, str] | None = None, header_spec: HeaderSpec | None = None) -> None:
        """Turn a reservation into usage. counted=False: the provider never accepted the request."""
        if res in self._inflight[res.pool]:
            self._inflight[res.pool].remove(res)
        if not counted:
            if res.entry in self._minute[res.pool]:
                self._minute[res.pool].remove(res.entry)
            return
        real = tokens if tokens is not None else res.est_tokens
        res.entry[1] = float(real)  # the minute window gets the real size too
        self.store.add_usage(res.pool, res.day, requests=1, tokens=real,
                             estimated_tokens=0 if tokens is not None else real)
        if headers and header_spec:
            self._sync_from_headers(res, headers, header_spec)

    def _sync_from_headers(self, res: Reservation, headers: Mapping[str, str], spec: HeaderSpec) -> None:
        def number(name: str | None) -> int | None:
            value = headers.get(name) if name else None
            try:
                return int(float(value)) if value is not None else None
            except ValueError:
                return None

        limit = number(spec.limit_requests_day)
        remaining = number(spec.remaining_requests_day)
        if limit is not None and limit != self.pools[res.pool].requests_per_day:
            self._limit_override[res.pool] = limit
        rpd = self._limit_override.get(res.pool, self.pools[res.pool].requests_per_day)
        if remaining is not None and rpd:
            self.store.sync_requests(res.pool, res.day, max(0, rpd - remaining))

    def block_for_today(self, pool: str) -> None:
        self.store.block_for_day(pool, self.day(pool))


Notify = Callable[[dict[str, Any]], Awaitable[None]]


class Meter:
    """One chat's view of one pool: the tracker plus that chat's notifications and Stop button."""

    def __init__(self, tracker: QuotaTracker, pool: str, label: str, notify: Notify,
                 should_stop: Callable[[], bool], header_spec: HeaderSpec | None = None) -> None:
        self.tracker = tracker
        self.pool = pool
        self.label = label
        self.notify = notify
        self.should_stop = should_stop
        self.header_spec = header_spec

    async def acquire(self, est_tokens: int) -> Reservation:
        while True:
            wait = self.tracker.check(self.pool, self.label, est_tokens)
            if wait <= 0:
                return self.tracker.reserve(self.pool, est_tokens)
            await self.wait(wait, f"{self.label}'s per-minute limit")

    async def wait(self, seconds: float, reason: str) -> None:
        """Sleep, showing a countdown in the chat, and give up if the user presses Stop."""
        await self.notify({"type": "rate_wait", "model": self.label, "seconds": round(seconds), "reason": reason})
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            if self.should_stop():
                raise StopWaiting()
            await asyncio.sleep(min(0.5, left))

    async def release(self, res: Reservation, *, counted: bool, tokens: int | None = None,
                      headers: Mapping[str, str] | None = None) -> None:
        self.tracker.release(res, counted=counted, tokens=tokens, headers=headers, header_spec=self.header_spec)
        await self.notify({"type": "usage_changed"})

    async def exhausted_for_today(self) -> QuotaExhausted:
        self.tracker.block_for_today(self.pool)
        await self.notify({"type": "usage_changed"})
        return QuotaExhausted(
            f"{self.label} refused more requests today (its free daily limit is reached). It resets at "
            f"{self.tracker._reset_label(self.pool)}. Pick another model in the dropdown and say 'continue'."
        )
