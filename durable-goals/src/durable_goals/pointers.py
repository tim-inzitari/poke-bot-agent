from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import ResolutionError, ValidationError


_MISSING = object()


def _tokens(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValidationError(f"expected an RFC 6901 JSON pointer, got {pointer!r}")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def get_pointer(document: Any, pointer: str, default: Any = _MISSING) -> Any:
    current = document
    for token in _tokens(pointer):
        try:
            if isinstance(current, list):
                current = current[int(token)]
            else:
                current = current[token]
        except (KeyError, IndexError, TypeError, ValueError):
            if default is not _MISSING:
                return default
            raise ResolutionError(f"JSON pointer does not resolve: {pointer}") from None
    return current


def apply_operations(document: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    result = deepcopy(document)
    for operation in operations:
        op = operation.get("op")
        pointer = operation.get("path")
        if op not in {"set", "remove"}:
            raise ValidationError(f"unsupported amendment operation: {op!r}")
        tokens = _tokens(pointer)
        if not tokens:
            raise ValidationError("amendments may not replace or remove the document root")

        if "expect" in operation:
            actual = get_pointer(result, pointer, default=_MISSING)
            if actual is _MISSING or actual != operation["expect"]:
                raise ResolutionError(
                    f"amendment precondition failed at {pointer}: "
                    f"expected {operation['expect']!r}, observed "
                    f"{'<missing>' if actual is _MISSING else actual!r}"
                )

        parent: Any = result
        for token in tokens[:-1]:
            if isinstance(parent, list):
                try:
                    parent = parent[int(token)]
                except (IndexError, TypeError, ValueError) as exc:
                    raise ResolutionError(f"amendment parent does not resolve: {pointer}") from exc
            else:
                if token not in parent or not isinstance(parent[token], (dict, list)):
                    raise ResolutionError(f"amendment parent does not resolve: {pointer}")
                parent = parent[token]

        final = tokens[-1]
        if isinstance(parent, list):
            try:
                index = int(final)
            except ValueError as exc:
                raise ResolutionError(f"list amendment index is invalid: {pointer}") from exc
            if op == "set":
                if index == len(parent):
                    parent.append(deepcopy(operation.get("value")))
                elif 0 <= index < len(parent):
                    parent[index] = deepcopy(operation.get("value"))
                else:
                    raise ResolutionError(f"list amendment index is out of range: {pointer}")
            else:
                try:
                    del parent[index]
                except IndexError as exc:
                    raise ResolutionError(f"remove target does not exist: {pointer}") from exc
        elif op == "set":
            if "value" not in operation:
                raise ValidationError(f"set operation lacks value at {pointer}")
            parent[final] = deepcopy(operation["value"])
        else:
            if final not in parent:
                raise ResolutionError(f"remove target does not exist: {pointer}")
            del parent[final]
    return result
