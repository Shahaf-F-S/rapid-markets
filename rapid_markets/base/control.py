# control.py

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable
import warnings

from ccxt.base.errors import NetworkError


__all__ = [
    'Control'
]


def print_on_catch(retry: Control, exception: Exception):
    print(retry.on_catch_message(exception))


def warn_on_catch(retry: Control, exception: Exception):
    warnings.warn("\n" + retry.on_catch_message(exception), stacklevel=3)


@dataclass(kw_only=True)
class Control:

    max_fails: int = 1
    interval: dt.timedelta = field(default_factory=dt.timedelta)
    timeout: dt.timedelta | None = None
    name: str | None = None
    start: dt.datetime = field(default_factory=dt.datetime.now)
    catch: list[type[Exception]] = field(default_factory=list)
    on_catch: Callable[[Control, Exception], ...] | None = None
    times: list[dt.datetime] = field(default_factory=list)
    running: bool = True

    def on_catch_message(self: Control, exception: Exception):
        name = f'[{self.name}] ' if self.name else ''
        return f'{name}{type(exception).__name__}:\n\t{exception}'

    @classmethod
    def request_timeout(
        cls, /, *,
        max_fails: int,
        interval: dt.timedelta,
        name: str | None = None,
        timeout: dt.timedelta | None = None,
        on_catch: Callable[[Control, Exception], ...] | None = None
    ) -> Control:
        return cls(
            name=name, max_fails=max_fails, interval=interval,
            timeout=timeout, catch=[NetworkError], on_catch=on_catch
        )

    @classmethod
    def suppress_request_timeout(
        cls, /, *,
        max_fails: int,
        interval: dt.timedelta,
        name: str | None = None,
        timeout: dt.timedelta | None = None
    ) -> Control:
        return cls.request_timeout(
            max_fails=max_fails, interval=interval,
            name=name, timeout=timeout
        )

    @classmethod
    def print_request_timeout(
        cls, /, *,
        max_fails: int,
        interval: dt.timedelta,
        name: str | None = None,
        timeout: dt.timedelta | None = None
    ) -> Control:
        return cls.request_timeout(
            max_fails=max_fails, interval=interval,
            name=name, timeout=timeout, on_catch=print_on_catch
        )

    @classmethod
    def warn_request_timeout(
        cls, /, *,
        max_fails: int,
        interval: dt.timedelta,
        name: str | None = None,
        timeout: dt.timedelta | None = None
    ) -> Control:
        return cls.request_timeout(
            max_fails=max_fails, interval=interval,
            name=name, timeout=timeout, on_catch=warn_on_catch
        )

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not isinstance(exc_val, tuple(self.catch)):
            return False

        if self.on_catch:
            self.on_catch(self, exc_val)

        self.step()

        if not self.run():
            return False

        return True

    def run(self) -> bool:
        if (not self.running) or (
            (self.timeout and dt.datetime.now() > (self.start + self.timeout)) or
            (len(self.times) == self.max_fails)
        ):
            self.stop()

            return False

        return True

    def stop(self):
        self.running = False

    def step(self):
        t = dt.datetime.now()
        times = self.times

        if times and t - times[-1] < self.interval:
            times.append(t)

        elif times:
            times.pop(-1)

    def copy(self, name: str, *, state: bool = False, deep: bool = False) -> Control:
        if not state:
            return Control(
                name=name,
                timeout=self.timeout,
                max_fails=self.max_fails,
                interval=self.interval,
                catch=self.catch.copy() if deep else self.catch,
                on_catch=self.on_catch
            )

        return Control(
            name=name,
            timeout=self.timeout,
            max_fails=self.max_fails,
            interval=self.interval,
            start=self.start,
            times=self.times.copy() if deep else self.times,
            catch=self.catch.copy() if deep else self.catch,
            on_catch=self.on_catch,
            running=self.running
        )
