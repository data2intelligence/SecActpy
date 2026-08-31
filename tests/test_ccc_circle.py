"""Tests for secactpy.visualization.ccc_circle (CCC chord diagram).

Python port of R SecAct's ``SecAct.CCC.circle``. The property worth protecting is
the contrast one: when two conditions are drawn side by side they must share a
single geometry, so a chord that is thinner means fewer edges rather than a
differently-scaled circle.
"""

import numpy as np
import pandas as pd
import pytest

from secactpy import ccc_circle

CT = ["T/NK cell", "B cell", "Myeloid cell"]


def _edges(n, seed):
    r = np.random.RandomState(seed)
    return pd.DataFrame({"sender": r.choice(CT, n), "receiver": r.choice(CT, n),
                         "secretedProtein": [f"SP{i % 9}" for i in range(n)]})


def _arc_starts(fig):
    """First vertex of each node arc, keyed by cell type, per panel."""
    out = {}
    for tr in fig.data:
        t = tr.text
        if isinstance(t, str) and "->" not in t and "<br>" in t:
            out.setdefault(t.split("<br>")[0], []).append(
                (round(float(tr.x[0]), 9), round(float(tr.y[0]), 9)))
    return out


def test_empty_input_returns_a_figure_not_an_error():
    fig = ccc_circle(pd.DataFrame(columns=["sender", "receiver"]))
    assert fig is not None and len(fig.data) == 0


def test_single_panel_draws_arcs_and_chords():
    fig = ccc_circle(_edges(200, 0), title="t")
    assert len(fig.data) > len(CT), "no chords drawn, only arcs"
    assert {c for c in _arc_starts(fig)} == set(CT)


def test_contrast_panels_share_one_geometry():
    """The claim the docstring makes, and the reason the contrast is readable.

    Sizing each panel from its own totals would rescale the arcs independently,
    so a cell type sending fewer edges would render as a smaller NODE rather than
    as the same node with thinner chords -- which inverts what the reader is
    meant to compare.
    """
    fig = ccc_circle(_edges(220, 1), compare_to=_edges(60, 2),
                     labels=("Responder", "Non-responder"))
    starts = _arc_starts(fig)
    for ct, pts in starts.items():
        assert len(pts) == 2, f"{ct} not drawn in both panels"
        assert pts[0] == pts[1], f"{ct} arc differs between panels: {pts}"


def test_contrast_chords_are_thinner_where_there_are_fewer_edges():
    """Same geometry, fewer edges -> less total chord area in that panel."""
    big, small = _edges(400, 3), _edges(80, 4)
    fig = ccc_circle(big, compare_to=small)
    # chord traces carry "a -> b: n edges"; sum the counts per panel
    tot = [0, 0]
    half = len(fig.data) // 2
    for i, tr in enumerate(fig.data):
        if isinstance(tr.text, str) and "->" in tr.text:
            tot[0 if i < half else 1] += int(tr.text.rsplit(": ", 1)[1].split()[0])
    assert tot[0] > tot[1], f"panel totals {tot} do not reflect edge counts"


def test_sender_receiver_filter_keeps_geometry_and_hides_chords():
    """R's sender/receiver blank the chords without moving the arcs, so the
    filtered view is comparable to the unfiltered one."""
    e = _edges(240, 5)
    full = ccc_circle(e)
    filt = ccc_circle(e, sender="B cell")
    assert _arc_starts(full) == _arc_starts(filt), "filtering moved the arcs"
    hidden = [tr for tr in filt.data
              if isinstance(tr.text, str) and "->" in tr.text and tr.opacity == 0.0]
    assert hidden, "sender filter hid nothing"
    kept = [tr for tr in filt.data
            if isinstance(tr.text, str) and tr.text.startswith("B cell ->")]
    assert all(tr.opacity > 0 for tr in kept), "sender filter hid its own sender"


def test_accepts_the_full_result_dict():
    e = _edges(120, 6)
    a = ccc_circle(e)
    b = ccc_circle({"secreted_protein_ccc": e})
    assert len(a.data) == len(b.data)


def test_colors_accept_dict_and_list():
    e = _edges(120, 7)
    d = ccc_circle(e, colors={"B cell": "#123456"})
    assert any(getattr(tr, "fillcolor", None) == "#123456" for tr in d.data)
    lst = ccc_circle(e, colors=["#111111", "#222222", "#333333"])
    fills = {getattr(tr, "fillcolor", None) for tr in lst.data}
    assert "#111111" in fills
