"""The usage counter (quota.py) with a fake clock."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.config import PoolSpec
from app.quota import HeaderSpec, QuotaExhausted, QuotaTracker
from app.store import Store

PACIFIC = ZoneInfo("America/Los_Angeles")


class Clock:
    def __init__(self, when: datetime) -> None:
        self.now = when.timestamp()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return Clock(datetime(2026, 10, 5, 12, 0, tzinfo=PACIFIC))


def tracker(clock, store=None, **pool):
    spec = PoolSpec(**{"requests_per_day": 10, "reset_timezone": "America/Los_Angeles", **pool})
    return QuotaTracker(store or Store("sqlite://"), {"p": spec}, reserve_calls=2, clock=clock)


def use(t, n, tokens=1000):
    for _ in range(n):
        assert t.check("p", "Test model", tokens) == 0
        t.release(t.reserve("p", tokens), counted=True, tokens=tokens)


def test_counts_requests_and_reported_tokens(clock):
    t = tracker(clock, tokens_per_day=100_000)
    use(t, 3, tokens=1234)
    snap = t.snapshot("p")
    assert (snap["requests_used"], snap["tokens_used"], snap["tokens_estimated"]) == (3, 3702, 0)
    assert snap["requests_left"] == 7 and snap["average_tokens_per_call"] == 1234


def test_requests_in_flight_count_as_used(clock):
    t = tracker(clock)
    res = t.reserve("p", 500)
    assert t.snapshot("p")["requests_used"] == 1  # a second chat can't take this call too
    t.release(res, counted=False)  # e.g. a 429: the provider didn't count it
    assert t.snapshot("p")["requests_used"] == 0


def test_missing_usage_is_estimated_and_marked(clock):
    t = tracker(clock)
    t.release(t.reserve("p", 700), counted=True, tokens=None)
    snap = t.snapshot("p")
    assert snap["tokens_used"] == 700 and snap["tokens_estimated"] == 700


def test_calls_left_is_the_tighter_of_requests_and_tokens(clock):
    t = tracker(clock, requests_per_day=100, tokens_per_day=20_000)
    use(t, 3, tokens=4000)  # average 4,000 per call; 8,000 tokens left -> 2 calls
    snap = t.snapshot("p")
    assert snap["requests_left"] == 97 and snap["tokens_left"] == 8000 and snap["calls_left"] == 2


def test_stops_when_only_the_reserve_is_left(clock):
    t = tracker(clock)  # 10 a day, reserve 2
    use(t, 7)
    assert t.usable("p") == (True, "")
    use(t, 1)  # 2 left
    usable, reason = t.usable("p")
    assert not usable and reason.startswith("2 calls left, kept in reserve")
    with pytest.raises(QuotaExhausted, match="down to about 2 free calls"):
        t.check("p", "Test model", 1000)


def test_the_day_resets_at_midnight_in_the_providers_time_zone(clock):
    t = tracker(clock)
    use(t, 8)
    assert not t.usable("p")[0]
    clock.advance(11 * 3600 + 59 * 60)  # 23:59 Pacific: still the same provider day
    assert not t.usable("p")[0]
    clock.advance(120)  # 00:01 Pacific: a new day
    assert t.usable("p")[0] and t.snapshot("p")["requests_used"] == 0


def test_counts_survive_a_restart(clock):
    store = Store("sqlite://")
    use(tracker(clock, store), 4)
    assert tracker(clock, store).snapshot("p")["requests_used"] == 4


def test_the_providers_own_count_wins(clock):
    t = tracker(clock, requests_per_day=1000)
    spec = HeaderSpec("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests")
    res = t.reserve("p", 100)
    t.release(res, counted=True, tokens=100,
              headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "957"}, header_spec=spec)
    snap = t.snapshot("p")
    assert snap["requests_used"] == 43 and snap["synced_at"] is not None  # e.g. used from another app


def test_a_different_limit_reported_by_the_provider_replaces_ours(clock):
    t = tracker(clock, requests_per_day=1000)
    spec = HeaderSpec("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests")
    t.release(t.reserve("p", 1), counted=True, tokens=1,
              headers={"x-ratelimit-limit-requests": "14400", "x-ratelimit-remaining-requests": "14399"}, header_spec=spec)
    assert t.snapshot("p")["requests_limit"] == 14400 and t.snapshot("p")["requests_used"] == 1


def test_a_daily_refusal_blocks_until_the_reset(clock):
    t = tracker(clock)
    t.block_for_today("p")
    with pytest.raises(QuotaExhausted, match="used up its free requests"):
        t.check("p", "Test model", 10)
    assert t.snapshot("p")["calls_left"] == 0
    clock.advance(13 * 3600)
    assert t.usable("p")[0]


def test_per_minute_request_limit_means_waiting_not_stopping(clock):
    t = tracker(clock, requests_per_day=None, requests_per_minute=2)
    use(t, 2)
    wait = t.check("p", "Test model", 10)
    assert 59 <= wait <= 60
    clock.advance(61)
    assert t.check("p", "Test model", 10) == 0


def test_per_minute_token_limit_waits_for_enough_tokens_to_expire(clock):
    t = tracker(clock, requests_per_day=None, tokens_per_minute=8000)
    use(t, 1, tokens=5000)
    clock.advance(10)
    use(t, 1, tokens=2000)
    assert 49 <= t.check("p", "Test model", 3000) <= 50  # the 5,000 from 10 s ago must expire first
    with pytest.raises(QuotaExhausted, match="more than Test model's limit of 8,000 tokens per minute"):
        t.check("p", "Test model", 9000)


def test_no_daily_limit_means_no_calls_left_figure(clock):
    t = tracker(clock, requests_per_day=None, requests_per_minute=40)
    use(t, 5)
    snap = t.snapshot("p")
    assert snap["calls_left"] is None and snap["requests_used"] == 5 and t.usable("p")[0]
