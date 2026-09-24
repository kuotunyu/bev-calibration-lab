"""A number quoted in prose must be the rounded value of the evidence it names."""

from __future__ import annotations

import pytest

EN_DASH = chr(0x2013)  # ranges in prose use an en dash, which is not a minus sign
VALUES = {
    ("metrics", "/runs/identity/roll:1/pixel"): 22.444,
    ("metrics", "/runs/identity/pitch:1/pixel"): 23.95468,
    ("metrics", "/scenes"): 30,
    ("metrics", "/tiny"): 1.4487761754180893e-11,
    ("metrics", "/negative"): -10.554,
    ("metrics", "/flag"): True,
    ("metrics", "/runs/a~1b/x:0/pixel"): 1.5,
    ("envelope", "/count"): 60,
}


def resolve(document: str, pointer: str) -> int | float:
    return VALUES[(document, pointer)]


def audit(text: str):  # type: ignore[no-untyped-def]
    from bevcalib.report.document_bindings import audit_markdown

    return audit_markdown(text, resolve)


def test_bound_numbers_that_match_their_rounded_evidence_pass() -> None:
    bindings, violations = audit(
        f"Shifts of 22.44{EN_DASH}23.95 px on 30 scenes, in all 60. "
        "<!-- bind: 22.44 = metrics#/runs/identity/roll:1/pixel ; "
        "23.95 = metrics#/runs/identity/pitch:1/pixel ; 30 = metrics#/scenes ; "
        "60 = envelope#/count -->\n"
        "A tiny error rounds to 0.00 and a change of -10.55 px. "
        "<!-- bind: 0.00 = metrics#/tiny ; -10.55 = metrics#/negative -->"
    )

    assert violations == ()
    assert [(item.line, item.display) for item in bindings] == [
        (1, "22.44"),
        (1, "23.95"),
        (1, "30"),
        (1, "60"),
        (2, "0.00"),
        (2, "-10.55"),
    ]
    assert bindings[0].source == "metrics#/runs/identity/roll:1/pixel"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Shift 22.45 px. <!-- bind: 22.45 = metrics#/runs/identity/roll:1/pixel -->",
            "does not match",
        ),
        (
            "Shift 22.5 px. <!-- bind: 22.5 = metrics#/runs/identity/roll:1/pixel -->",
            "does not match",
        ),
        ("Shift 30 px. <!-- bind: 30 = metrics#/flag -->", "does not match"),
        (
            "Shift 122.44 px. <!-- bind: 22.44 = metrics#/runs/identity/roll:1/pixel -->",
            "not shown",
        ),
        ("Change 10.55 px. <!-- bind: -10.55 = metrics#/negative -->", "not shown"),
        ("Change of --10.55. <!-- bind: 10.55 = metrics#/negative -->", "does not match"),
        ("Shift 22.44 px. <!-- bind: 22.44 = metrics#/missing -->", "cannot be resolved"),
        ("Shift 22.44 px. <!-- bind: 22.44 = metrics/runs -->", "cannot be resolved"),
        ("Shift 22.44 px. <!-- bind: 22.44 = #/runs -->", "cannot be resolved"),
        ("Shift 22.44 px. <!-- bind: 22.44 = metrics#runs -->", "cannot be resolved"),
        ("About 22 px. <!-- bind: ~22 = metrics#/scenes -->", "not a plain number"),
        ("Shift 22.44 px. <!-- bind: 22.44 -->", "has no '='"),
        ("Nothing here. <!-- bind: -->", "empty bind marker"),
        (
            "<!-- bind: 30 = metrics#/scenes --> The number follows 30 the marker.",
            "not shown",
        ),
    ],
)
def test_every_way_a_bound_number_can_drift_is_reported(text: str, expected: str) -> None:
    bindings, violations = audit(text)

    assert bindings == ()
    assert len(violations) == 1
    assert violations[0].startswith("line 1: ")
    assert expected in violations[0]


def test_a_table_is_bound_cell_by_cell_from_its_backticked_keys() -> None:
    bindings, violations = audit(
        "Intro\n"
        "<!-- bind-table: metrics#/runs/{column}/{row}/pixel -->\n"
        "| Fault | `identity` |\n"
        "| --- | ---: |\n"
        "| tilt `roll:1` | 22.44 |\n"
        "| pan `pitch:1` | 23.95 |\n"
        "\n"
        "<!-- bind-table: metrics#/runs/{column}/{row}/pixel -->\n"
        "| Fault | `a/b` |\n"
        "| --- | ---: |\n"
        "| zero `x:0` | 1.5 |\n"
    )

    assert violations == ()
    assert [(item.line, item.display, item.source) for item in bindings] == [
        (5, "22.44", "metrics#/runs/identity/roll:1/pixel"),
        (6, "23.95", "metrics#/runs/identity/pitch:1/pixel"),
        (11, "1.5", "metrics#/runs/a~1b/x:0/pixel"),
    ]


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            "| Fault | `identity` |\n| --- | --- |\n| tilt `roll:1` | 22.43 |\n",
            "line 4: 22.43 does not match",
        ),
        ("| Fault | `identity` |\n| --- | --- |\n| tilt roll:1 | 22.44 |\n", "line 4: table row"),
        ("| Fault | identity |\n| --- | --- |\n| tilt `roll:1` | 22.44 |\n", "line 4: table row"),
        (
            "| Fault | `identity` |\n| --- | --- |\n| tilt `roll:1` | 22.44 | 1 |\n",
            "line 4: table row",
        ),
        ("| Fault | `identity` |\n| --- | --- |\n", "line 1: bind-table is not followed"),
        ("No table here.\n", "line 1: bind-table is not followed"),
    ],
)
def test_a_malformed_or_drifted_bound_table_is_reported(table: str, expected: str) -> None:
    _, violations = audit("<!-- bind-table: metrics#/runs/{column}/{row}/pixel -->\n" + table)

    assert len(violations) == 1
    assert violations[0].startswith(expected)


def test_documents_are_audited_together_and_violations_name_their_document() -> None:
    from bevcalib.report.document_bindings import audit_markdown_documents

    bindings, violations = audit_markdown_documents(
        [
            ("README.md", "30 scenes <!-- bind: 30 = metrics#/scenes -->"),
            ("docs/errata.md", "31 scenes <!-- bind: 31 = metrics#/scenes -->"),
        ],
        resolve,
    )

    assert [item.display for item in bindings] == ["30"]
    assert violations == ("docs/errata.md: line 1: 31 does not match metrics#/scenes = 30",)


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [(22.444, 2, "22.44"), (0.625, 2, "0.62"), (0.635, 2, "0.64"), (1.0, 0, "1"), (60, 0, "60")],
)
def test_display_rounding_uses_the_exact_decimal_form_of_the_value(
    value: float, decimals: int, expected: str
) -> None:
    from bevcalib.report.document_bindings import format_bound

    assert format_bound(value, decimals) == expected
