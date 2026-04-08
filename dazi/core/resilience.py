"""Resilience — retry, circuit breaker, abort signal."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# ─────────────────────────────────────────────────────────
# ABORT SIGNAL
# ─────────────────────────────────────────────────────────

class AbortSignal:
    """Cooperative cancellation signal.

    Uses threading.Event which works for both sync and async contexts.
    """

    def __init__(self):
        self._event = threading.Event()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def abort(self) -> None:
        """Signal cancellation."""
        self._event.set()

    def check(self) -> None:
        """Raise if aborted. Call in long-running operations."""
        if self.aborted:
            raise AbortError("Operation aborted")

    async def async_check(self) -> None:
        """Async version of check()."""
        if self.aborted:
            raise AbortError("Operation aborted")
        # Yield to event loop to allow abort to propagate
        await asyncio.sleep(0)


class AbortError(Exception):
    """Raised when an operation is cancelled via AbortSignal."""
    pass


# ─────────────────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for API calls.

    States:
      CLOSED:   Normal operation — failures increment counter
      OPEN:     Failing — all calls immediately rejected
      HALF_OPEN: Probing — allow one call to test if service recovered
    """
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    _failure_count: int = field(default=0, init=False, repr=False)
    _open_until: float = field(default=0.0, init=False, repr=False)

    @property
    def state(self) -> str:
        """Current state: closed, open, or half_open."""
        if self._failure_count >= self.failure_threshold:
            if time.time() >= self._open_until:
                return "half_open"
            return "open"
        return "closed"

    def record_success(self) -> None:
        """Record a successful call — reset failure counter."""
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call — may trip the circuit."""
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._open_until = time.time() + self.cooldown_seconds

    def allow_request(self) -> bool:
        """Should we attempt a request?"""
        state = self.state
        if state == "closed":
            return True
        if state == "half_open":
            return True  # Allow one probe request
        return False  # open — reject

    async def wait_if_open(self) -> None:
        """If open, wait until cooldown expires (for async contexts)."""
        if self.state == "open":
            wait_time = self._open_until - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)


# ─────────────────────────────────────────────────────────
# RETRY POLICY
# ─────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    # Error types that should trigger retry
    retryable_errors: tuple[type, ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


def _calculate_delay(attempt: int, policy: RetryPolicy) -> float:
    """Calculate exponential backoff delay with optional jitter.

    Formula:
      delay = min(base_delay * 2^attempt, max_delay)
      + random jitter (0 to base_delay)
    """
    delay = min(policy.base_delay * (2 ** attempt), policy.max_delay)
    if policy.jitter:
        delay += random.uniform(0, policy.base_delay)
    return delay


def _is_retryable(error: Exception, policy: RetryPolicy) -> bool:
    """Check if an error is worth retrying.

    Specific status code handling:
      - 529: overloaded
      - 400: context overflow
      - 429: rate limited
    """
    # Check explicit retryable types
    if isinstance(error, policy.retryable_errors):
        return True

    # Check error message for API status codes
    error_str = str(error).lower()
    if "429" in error_str or "rate limit" in error_str:
        return True
    if "529" in error_str or "overloaded" in error_str:
        return True
    if "503" in error_str or "service unavailable" in error_str:
        return True
    if "500" in error_str or "internal server error" in error_str:
        return True
    if "connection" in error_str or "timeout" in error_str:
        return True

    return False


async def with_retry(
    func: Callable[..., Any],
    policy: RetryPolicy | None = None,
    circuit: CircuitBreaker | None = None,
    abort: AbortSignal | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a function with retry, circuit breaker, and abort support.

    Args:
        func: The async function to execute.
        policy: Retry configuration. Uses defaults if None.
        circuit: Optional circuit breaker.
        abort: Optional abort signal.
        **kwargs: Arguments to pass to func.

    Returns:
        The result of func.

    Raises:
        AbortError: If abort signal fires.
        CircuitOpenError: If circuit breaker is open.
        MaxRetriesError: If all retries exhausted.
    """
    if policy is None:
        policy = RetryPolicy()

    last_error: Exception | None = None

    for attempt in range(policy.max_retries + 1):
        # Check abort
        if abort and abort.aborted:
            raise AbortError("Operation aborted before attempt")

        # Check circuit breaker
        if circuit and not circuit.allow_request():
            if attempt == 0:
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN. {circuit.failure_threshold} consecutive failures. "
                    f"Cooldown: {circuit.cooldown_seconds}s"
                )
            # Wait for cooldown on retry
            if circuit:
                await circuit.wait_if_open()

        try:
            result = await func(**kwargs) if asyncio.iscoroutinefunction(func) else func(**kwargs)
            if circuit:
                circuit.record_success()
            return result

        except AbortError:
            raise

        except Exception as e:
            last_error = e

            if circuit:
                circuit.record_failure()

            if abort and abort.aborted:
                raise AbortError("Operation aborted during retry")

            if attempt < policy.max_retries and _is_retryable(e, policy):
                delay = _calculate_delay(attempt, policy)
                # Log retry (in production, this would use proper logging)
                print(f"  [retry] Attempt {attempt + 1}/{policy.max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                continue

            # Not retryable or out of retries
            raise MaxRetriesError(
                f"Failed after {attempt + 1} attempts: {e}"
            ) from e

    raise MaxRetriesError(f"Failed after {policy.max_retries} attempts: {last_error}")


class CircuitOpenError(Exception):
    """Raised when circuit breaker rejects a request."""
    pass


class MaxRetriesError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass
