# callbacks.py

import asyncio
from abc import abstractmethod, ABC
from collections.abc import Callable
from inspect import iscoroutinefunction
from typing import Self, Awaitable


__all__ = [
    'BaseCallbacks',
    'AsyncGatherCallbacks',
    'AsyncIterativeCallbacks',
    'IterativeCallbacks',
    'MixIterativeCallbacks',
    'BaseAsyncCallbacks',
    'run_task'
]


class BaseCallbacks[In, Out](list, ABC):

    def __bool__(self) -> bool:
        return True

    @staticmethod
    def _args(data: In = ...) -> tuple:
        return (data,) if data is not Ellipsis else ()

    def __call__(self, data: In = ...) -> Out:
        return self.call(self._args(data))

    @abstractmethod
    def call(self, args: tuple) -> Out:
        pass

    def collect(self, *calls: Callable) -> Self:
        self.extend(calls)
        return self


class IterativeCallbacks[In, Out](BaseCallbacks[In, Out]):

    def call(self, args: tuple):
        for call in self:
            call(*args)


class BaseAsyncCallbacks[In, Out](BaseCallbacks[In, Out], ABC):

    async def __call__(self, data: In = ...) -> Out:
        return await self.call(self._args(data))

    @abstractmethod
    async def call(self, args: tuple) -> Out:
        pass


class AsyncIterativeCallbacks[In, Out](BaseAsyncCallbacks[In, Out]):

    async def call(self, args: tuple) -> Out:
        for call in self:
            await call(*args)


class MixIterativeCallbacks[In, Out](BaseAsyncCallbacks[In, Out]):

    async def call(self, args: tuple):
        for call in self:
            if iscoroutinefunction(call):
                await call(*args)

            else:
                call(*args)


class AsyncGatherCallbacks[In, Out](BaseAsyncCallbacks[In, Out]):

    async def call(self, args: tuple) -> Out:
        await asyncio.gather(*[call(*args) for call in self])


async def run_task(
    init: Awaitable | None = None,
    background: Awaitable | list[Awaitable] | None = None,
    task: Awaitable | None = None,
    finish: Awaitable | list[Awaitable] | None = None
):
    if init is not None:
        await init

    background_task = None
    finish_task = None

    if background is not None:
        if isinstance(background, list):
            background_task = asyncio.gather(*background)

        else:
            # noinspection PyTypeChecker
            background_task = asyncio.create_task(background)

    try:
        if task is not None:
            await task

    finally:
        if finish is not None:
            if isinstance(finish, list):
                finish_task = asyncio.gather(*finish)

            else:
                # noinspection PyTypeChecker
                finish_task = asyncio.create_task(finish)

        if background_task is not None:
            background_task.cancel()
            await asyncio.gather(background_task, return_exceptions=True)

        if finish_task is not None:
            finish_task.cancel()
            await asyncio.gather(finish_task, return_exceptions=True)