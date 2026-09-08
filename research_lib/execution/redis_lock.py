"""Redis distributed locking with ownership-checked release (Phase 31).

Pattern (Redis docs single-instance recommendation):
  acquire: SET resource_name random_value NX PX 30000
  release: Lua compare-and-delete (never unconditional DEL)
  verify:  ownership before release using the unique random value

The lock owner is uniquely identified by a random token. A lock is NEVER released with an
unconditional DEL. All interactions go through the redis-py client interface so the same
code is unit-tested against fakeredis and usable against a real Redis.

Also provides a Redis-backed idempotency guard: SETNX on the idempotency key with a TTL, so
among multiple workers receiving the same signal only one proceeds to submit the order.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Protocol


class RedisLike(Protocol):
    """Minimal redis client interface used by the lock (satisfied by redis.Redis and
    fakeredis.FakeRedis)."""

    def set(self, name, value, nx=False, px=None, ex=None): ...

    def get(self, name): ...

    def delete(self, *names): ...

    def eval(self, script, numkeys, *keys_and_args): ...


_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

_VERIFY_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return 1
else
    return 0
end
"""


@dataclass
class DistributedLock:
    client: RedisLike
    resource: str
    ttl_ms: int = 30_000
    token: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def _key(self) -> str:
        return f"lock:{self.resource}"

    def acquire(self) -> bool:
        """SET resource random_value NX PX ttl - only succeeds if not already held."""
        return bool(self.client.set(self._key, self.token, nx=True, px=self.ttl_ms))

    def is_owner(self) -> bool:
        """Verify ownership (current value == our token)."""
        return self._owned_bytes()

    def release(self) -> bool:
        """Ownership-checked release via Lua compare-and-delete (never unconditional DEL)."""
        try:
            res = self.client.eval(_RELEASE_SCRIPT, 1, self._key, self.token)
            return bool(res)
        except Exception:
            # If the Lua path fails we must NOT fall back to DEL (would break ownership).
            # Re-verify first; only delete if we still own the lock.
            if self._owned_bytes():
                return bool(self.client.delete(self._key))
            return False

    def _owned_bytes(self) -> bool:
        val = self.client.get(self._key)
        if isinstance(val, bytes):
            return val.decode() == self.token
        return val == self.token


@dataclass
class IdempotencyGuard:
    """SETNX guard so only one worker submits an order for a given idempotency key."""

    client: RedisLike
    ttl_ms: int = 60_000

    def try_claim(self, idempotency_key: str) -> bool:
        return bool(self.client.set(f"idem:{idempotency_key}", "1", nx=True, px=self.ttl_ms))

    def release(self, idempotency_key: str) -> None:
        self.client.delete(f"idem:{idempotency_key}")
