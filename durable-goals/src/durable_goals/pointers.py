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
    tokens: list[str] = []
    for raw in pointer[1:].split("/"):
        index = 0
        while index < len(raw):
            if raw[index] == "~":
                if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                    raise ValidationError(
                        f"invalid RFC 6901 escape in JSON pointer: {pointer!r}"
                    )
                index += 2
            else:
                index += 1
        tokens.append(raw.replace("~1", "/").replace("~0", "~"))
    return tokens


def _array_index(token: str, *, pointer: str, allow_end: bool = False) -> int:
    if token == "-" and allow_end:
        return -1
    if not token or not token.isascii() or not token.isdigit():
        raise ResolutionError(f"list amendment index is invalid: {pointer}")
    if len(token) > 1 and token.startswith("0"):
        raise ResolutionError(f"list amendment index has a leading zero: {pointer}")
    return int(token)


def get_pointer(document: Any, pointer: str, default: Any = _MISSING) -> Any:
    current = document
    for token in _tokens(pointer):
        try:
            if isinstance(current, list):
                index = _array_index(token, pointer=pointer)
                current = current[index]
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
                    parent = parent[_array_index(token, pointer=pointer)]
                except (IndexError, TypeError, ValueError, ResolutionError) as exc:
                    raise ResolutionError(f"amendment parent does not resolve: {pointer}") from exc
            else:
                if token not in parent or not isinstance(parent[token], (dict, list)):
                    raise ResolutionError(f"amendment parent does not resolve: {pointer}")
                parent = parent[token]

        final = tokens[-1]
        if isinstance(parent, list):
            index = _array_index(final, pointer=pointer, allow_end=op == "set")
            if op == "set":
                if index == -1 or index == len(parent):
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
