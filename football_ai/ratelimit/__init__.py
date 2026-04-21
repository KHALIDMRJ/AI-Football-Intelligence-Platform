"""
Rate-limiting primitives — token bucket + FastAPI dependency factory.

Why a bucket (not a fixed window)
---------------------------------
Live-match traffic is bursty: a scout watching Liverpool–City will fire
off several requests when a goal is scored, then sit idle. A fixed
"60 requests per minute" window punishes that behaviour on bucket
boundaries. A token bucket lets short bursts through (up to ``capacity``)
while enforcing the long-run rate (``refill_per_second``), which matches
how users actually consume live analytics.

Why role-aware
--------------
A free_user polling /predictions/live every second is already trying to
replace a pro subscription. We throttle them tightly; pro_analyst /
club_scout get headroom; admin is unlimited.
"""

from .dependencies import RateLimitPolicy, rate_limit, role_rate_limit
from .token_bucket import TokenBucket

__all__ = [
    "TokenBucket",
    "RateLimitPolicy",
    "rate_limit",
    "role_rate_limit",
]
