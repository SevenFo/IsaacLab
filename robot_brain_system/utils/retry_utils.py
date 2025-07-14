import functools
import pickle
import time
from typing import Any, Callable, Optional, Tuple, Type


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay_seconds: float = 60.0,
    exceptions_to_retry: Tuple[Type[Exception], ...] = (
        pickle.UnpicklingError,
    ),  # Retry on specific exceptions
    retry_on_return_value_condition: Optional[
        Callable[[Any], bool]
    ] = None,  # Retry based on return value
    logger_func: Optional[Callable[[str], None]] = print,
):
    """
    A decorator to retry a function call on specific exceptions or return value conditions.

    Args:
        max_attempts: Maximum number of attempts.
        delay_seconds: Initial delay between retries in seconds.
        backoff_factor: Factor by which the delay increases after each attempt (e.g., 2 for exponential backoff).
        max_delay_seconds: Maximum delay between retries.
        exceptions_to_retry: A tuple of exception types that should trigger a retry.
        retry_on_return_value_condition: A callable that takes the function's return value
                                         and returns True if a retry is needed.
        logger_func: Function to use for logging retry attempts (e.g., print or a logger method).
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal delay_seconds  # Allow modification for backoff
            current_delay = delay_seconds
            last_exception: Optional[Exception] = None
            last_result: Any = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    last_result = result  # Store the last result

                    should_retry_based_on_value = False
                    if retry_on_return_value_condition:
                        if retry_on_return_value_condition(result):
                            should_retry_based_on_value = True

                    if not should_retry_based_on_value:
                        return result  # Success or non-retryable failure based on return value

                    # If we are here, it means retry_on_return_value_condition was True
                    if logger_func:
                        logger_func(
                            f"[Retry] Attempt {attempt}/{max_attempts} of '{func.__name__}' "
                            f"failed due to return value condition. Retrying in {current_delay:.2f}s..."
                        )

                except exceptions_to_retry as e:
                    last_exception = e
                    if logger_func:
                        logger_func(
                            f"[Retry] Attempt {attempt}/{max_attempts} of '{func.__name__}' "
                            f"failed with {type(e).__name__}: {e}. Retrying in {current_delay:.2f}s..."
                        )
                except (
                    Exception
                ) as e:  # Catch other exceptions not in exceptions_to_retry
                    if logger_func:
                        logger_func(
                            f"[Retry] Unhandled exception {type(e).__name__} in '{func.__name__}' "
                            f"on attempt {attempt}. Not retrying this exception. Re-raising."
                        )
                    raise  # Re-raise unhandled exceptions immediately

                if attempt < max_attempts:
                    time.sleep(current_delay)
                    current_delay = min(
                        current_delay * backoff_factor, max_delay_seconds
                    )
                else:
                    # All attempts failed
                    if logger_func:
                        if last_exception:
                            logger_func(
                                f"[Retry] All {max_attempts} attempts of '{func.__name__}' failed. "
                                f"Last exception: {type(last_exception).__name__}: {last_exception}"
                            )
                            raise last_exception  # Re-raise the last caught retryable exception
                        elif should_retry_based_on_value:  # type: ignore
                            logger_func(
                                f"[Retry] All {max_attempts} attempts of '{func.__name__}' failed "
                                f"due to return value condition. Returning last result."
                            )
                            return last_result  # Return the last failing result
            return last_result  # Should ideally be covered by above, but as a fallback

        return wrapper

    return decorator
