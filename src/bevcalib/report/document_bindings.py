"""Check that every number bound in a Markdown document equals its evidence.

A README sentence carries an invisible marker naming where each of its numbers
comes from, with several items separated by semicolons::

    ... moves projected points by 22.44 px. <!-- bind: 22.44 = metrics#/runs/identity/roll:1/pixel_frame_p50_px/value -->

and a table is preceded by one marker whose pointer template is filled from the
backticked keys in its header row (``{column}``) and first column (``{row}``)::

    <!-- bind-table: metrics#/runs/{column}/{row}/pixel_frame_p50_px/value -->

The displayed text must be the source value rounded to exactly the decimals shown,
and must appear on the marked line. Resolution is supplied by the caller, so the
same check serves the claim-bound formal documents and derived analyses.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal

Resolver = Callable[[str, str], int | float]

_BIND = re.compile(r"<!--\s*bind:(?P<body>.*?)-->")
_TABLE = re.compile(r"^\s*<!--\s*bind-table:\s*(?P<source>\S+)\s*-->\s*$")
_KEY = re.compile(r"`([^`]+)`")
_DISPLAY = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class Binding:
    """One displayed number and the evidence location it was checked against."""

    line: int
    display: str
    source: str


def _source(reference: str) -> tuple[str, str]:
    document, separator, pointer = reference.partition("#")
    if not separator or not document or not pointer.startswith("/"):
        raise ValueError(f"malformed evidence reference: {reference!r}")
    return document, pointer


def format_bound(value: int | float, decimals: int) -> str:
    """Round the exact decimal form of a source value half-to-even for display."""

    return f"{Decimal(repr(value)):.{decimals}f}"


def _formatted(display: str, value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return format_bound(value, len(display.partition(".")[2])) == display


def _shown(display: str, text: str) -> bool:
    pattern = rf"(?<![\w.-]){re.escape(display)}(?!\d|\.\d)"
    return re.search(pattern, text) is not None


def _check(
    display: str, reference: str, text: str, line: int, resolve: Resolver
) -> tuple[Binding | None, str | None]:
    if not _DISPLAY.match(display):
        return None, f"line {line}: bound text is not a plain number: {display!r}"
    try:
        document, pointer = _source(reference)
        value = resolve(document, pointer)
    except (LookupError, ValueError) as exc:
        return None, f"line {line}: {display} cannot be resolved: {exc}"
    if not _formatted(display, value):
        return None, f"line {line}: {display} does not match {reference} = {value!r}"
    if not _shown(display, text):
        return None, f"line {line}: {display} is bound but not shown on the line"
    return Binding(line, display, reference), None


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def audit_markdown(text: str, resolve: Resolver) -> tuple[tuple[Binding, ...], tuple[str, ...]]:
    """Return every checked binding and every violation found in one document."""

    bindings: list[Binding] = []
    violations: list[str] = []
    lines = text.splitlines()

    def record(result: tuple[Binding | None, str | None]) -> None:
        binding, violation = result
        if binding is not None:
            bindings.append(binding)
        if violation is not None:
            violations.append(violation)

    for index, line in enumerate(lines):
        number = index + 1
        table = _TABLE.match(line)
        if table is not None:
            rows = []
            for candidate in lines[index + 1 :]:
                if not candidate.lstrip().startswith("|"):
                    break
                rows.append(candidate)
            if len(rows) < 3:
                violations.append(f"line {number}: bind-table is not followed by a table")
                continue
            columns = [_KEY.search(cell) for cell in _cells(rows[0])[1:]]
            for offset, row in enumerate(rows[2:], start=number + 3):
                cells = _cells(row)
                key = _KEY.search(cells[0])
                if key is None or len(cells) != len(columns) + 1 or None in columns:
                    violations.append(f"line {offset}: table row or header has no backticked key")
                    continue
                for column, cell in zip(columns, cells[1:], strict=True):
                    assert column is not None
                    reference = table.group("source").format(
                        column=_pointer_token(column.group(1)), row=_pointer_token(key.group(1))
                    )
                    record(_check(cell, reference, cell, offset, resolve))
            continue
        text_before = ""
        cursor = 0
        for match in _BIND.finditer(line):
            text_before += line[cursor : match.start()]
            cursor = match.end()
            items = [item for item in match.group("body").split(";") if item.strip()]
            if not items:
                violations.append(f"line {number}: empty bind marker")
            for item in items:
                display, separator, reference = item.partition("=")
                if not separator:
                    violations.append(f"line {number}: bind item has no '=': {item.strip()!r}")
                    continue
                record(_check(display.strip(), reference.strip(), text_before, number, resolve))
    return tuple(bindings), tuple(violations)


def audit_markdown_documents(
    documents: Iterable[tuple[str, str]], resolve: Resolver
) -> tuple[tuple[Binding, ...], tuple[str, ...]]:
    """Audit several named documents and prefix each violation with its name."""

    bindings: list[Binding] = []
    violations: list[str] = []
    for name, text in documents:
        found, failed = audit_markdown(text, resolve)
        bindings.extend(found)
        violations.extend(f"{name}: {violation}" for violation in failed)
    return tuple(bindings), tuple(violations)
