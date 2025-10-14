from __future__ import annotations
from typing import Callable, Dict, Any


class Registry:
    def __init__(self) -> None:
        self._fns: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., Any]):
            if name in self._fns:
                raise KeyError(f"{name} already registered")
            self._fns[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._fns:
            raise KeyError(f"{name} not found in registry")
        return self._fns[name]

    def names(self) -> list[str]:
        return sorted(self._fns.keys())


DATASETS = Registry()
METHODS = Registry()