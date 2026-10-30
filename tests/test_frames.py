import logging
import warnings

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ahl_npm import (
    ImpossibleInputWarning,
    MissingInputWarning,
    band,
    band_expr,
    frames,
    output_names,
    score,
    score_frame,
)


@pytest.fixture
def panels() -> pl.DataFrame:
    """Random panels per 100 g under made-up column names, with some values missing."""
    rng = np.random.default_rng(1)
    n = 2_000
    frame = pl.DataFrame(
        {
            "kcal_100g": rng.uniform(0, 900, n),
            "sat_100g": rng.uniform(0, 12, n),
            "sug_100g": rng.uniform(0, 50, n),
            "free_sug_100g": rng.uniform(0, 12, n),
            "salt_100g": rng.uniform(0, 3, n),
            "prot_100g": rng.uniform(0, 10, n),
            "fibre_100g": rng.uniform(0, 7, n),
            "fvn_share": rng.uniform(0, 1, n),
        }
    )
    return frame.with_columns(
        pl.when(pl.int_range(n) % 97 == 0).then(None).otherwise(pl.col("fibre_100g")).alias("fibre_100g")
    )


COLUMNS = {
    "energy_kcal": "kcal_100g",
    "satfat": "sat_100g",
    "sugars": "sug_100g",
    "free_sugars": "free_sug_100g",
    "salt": "salt_100g",
    "protein": "prot_100g",
    "fibre": "fibre_100g",
    "fvn_pct": pl.col("fvn_share") * 100,
    "fvns_pct": pl.col("fvn_share") * 100,
}


@pytest.mark.parametrize("scheme", ["2004", "2018"])
def test_frames_score_as_arrays_do(panels: pl.DataFrame, scheme: str) -> None:
    """A frame's scoring is the numpy scoring of its columns, nulls where NaN."""
    with pytest.warns(MissingInputWarning):
        scored = score_frame(panels, scheme, COLUMNS)
        arrays = score(
            scheme,
            **{
                name: panels.select(column if isinstance(column, pl.Expr) else pl.col(column)).to_series().to_numpy()
                for name, column in COLUMNS.items()
            },
        )
    for name in output_names(scheme):
        got = scored[f"{name}_{scheme}"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        np.testing.assert_array_equal(got, arrays[name])


def test_lazy_and_streaming_frames_score_as_eager_ones(panels: pl.DataFrame) -> None:
    """The same scoring, eager, lazy or streaming."""
    eager = score_frame(panels, "2018", COLUMNS, warn=False)
    lazy = score_frame(panels.lazy(), "2018", COLUMNS, warn=False)
    assert_frame_equal(lazy.collect(), eager)
    assert_frame_equal(lazy.collect(engine="streaming"), eager)


# A panel under the inputs' own names: 2004/05's worked example, 14 (and cNPM 42).
NAMED = {"energy_kj": [1500.0], "satfat": [4.5], "sugars": [20.0], "sodium": [0.5], "protein": [10.0], "fibre": [3.0]}


def test_columns_named_as_inputs_are_found_unless_columns_are_given() -> None:
    """Without `columns` a frame's inputs are its columns named as them; with it, only those it names, never a stray."""
    frame = pl.DataFrame({**NAMED, "fvn_points": [0], "salt": [900.0], "other_sodium": [None]})
    assert score_frame(frame.drop("salt", "other_sodium"), "2004")["npm_2004"].to_list() == [14]
    columns = {name: name for name in NAMED} | {"sodium": "other_sodium", "fvn_points": "fvn_points"}
    with pytest.warns(MissingInputWarning, match="sodium 1"):
        assert score_frame(frame, "2004", columns)["npm_2004"].to_list() == [None]


def test_one_mapping_serves_both_schemes() -> None:
    """A scheme leaves out the inputs it doesn't read, so a column only the other scheme reads need not exist."""
    frame = pl.DataFrame({**NAMED, "fvn_points": [0]})
    columns = {name: name for name in (*NAMED, "fvn_points")} | {"free_sugars": "no_such_column"}
    assert score_frame(frame, 2004, columns)["npm_2004"].to_list() == [14]


def test_outputs_choose_and_name_the_new_columns() -> None:
    """`outputs` picks which new columns to add, cNPM among them, and a mapping names them."""
    frame = pl.DataFrame({**NAMED, "fvn_points": [0]})
    picked = score_frame(frame, "2004", outputs=["npm", "cnpm"])
    assert picked.columns[len(frame.columns) :] == ["npm_2004", "cnpm_2004"]
    named = score_frame(frame, "2004", outputs={"sugars_points": "sugar_score", "npm": "npm_score"})
    assert named.select("sugar_score_2004", "npm_score_2004").row(0) == (4, 14)
    assert score_frame(frame, "2004", outputs="npm").columns[len(frame.columns) :] == ["npm_2004"]
    with pytest.raises(ValueError, match="free_sugars_points"):
        score_frame(frame, "2004", outputs=["free_sugars_points"])
    with pytest.raises(ValueError, match="share the names"):
        score_frame(frame, "2004", outputs={"npm": "score", "cnpm": "score"})


def test_new_columns_are_named_between_prefix_and_suffix() -> None:
    """`prefix` and `suffix` name the new columns, cNPM included."""
    frame = pl.DataFrame({**NAMED, "fvn_points": [0]})
    scored = score_frame(frame, "2004", prefix="old_", suffix="", cnpm=True)
    assert scored.columns[len(frame.columns) :] == [*(f"old_{name}" for name in output_names("2004")), "old_cnpm"]
    assert scored.select("old_npm", "old_cnpm").row(0) == (14, 42)


def test_scoring_never_replaces_an_input() -> None:
    """A new column that would take an input's name is an error, as is a column under the scoring's own."""
    frame = pl.DataFrame({**NAMED, "fvn_points": [0]})
    with pytest.raises(ValueError, match="fvn_points"):
        score_frame(frame, "2004", suffix="")
    with pytest.raises(ValueError, match="needs for itself"):
        score_frame(frame.with_columns(pl.lit(0).alias(frames._SCORING)), "2004")


def test_a_streaming_scoring_warns_once() -> None:
    """A streaming collect scores batches of rows at once, yet warns once, of the first batch it finds wanting."""
    n = 400_000
    frame = pl.LazyFrame({**{name: np.repeat(values, n) for name, values in NAMED.items()}, "fvn_points": 0})
    frame = frame.with_columns(pl.when(pl.int_range(pl.len()) % 2 == 0).then(pl.col("fibre")).alias("fibre"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score_frame(frame, "2004").collect(engine="streaming")
    assert [w.category for w in caught] == [MissingInputWarning]


def test_each_scoring_logs_what_it_reads_and_in_what_units(caplog: pytest.LogCaptureFixture) -> None:
    """The log says, per component, which column or expression it reads, in what units and how it converts."""
    frame = pl.DataFrame({"kcal": [358.5], "sat": [4.5], "sug": [20.0], "salt_g": [1.25], "fvn_share": [0.0]})
    columns = {"energy_kcal": "kcal", "satfat": "sat", "sugars": "sug", "salt": "salt_g", "protein": pl.lit(10.0)}
    columns |= {"fibre": pl.lit(3.0), "fvn_pct": pl.col("fvn_share") * 100}
    frames._LOGGED.clear()  # as in a fresh process
    with caplog.at_level(logging.INFO, logger="ahl_npm"):
        score_frame(frame, "2004", columns)
        score_frame(frame, "2004", columns)  # the same reading again, as in a sweep: not logged again
    assert caplog.text.count("NPM 2004 reads") == 1
    assert 'energy: energy_kcal <- "kcal", kcal x 4.184 to kJ' in caplog.text
    assert 'sodium: salt <- "salt_g", g / 2.5 to sodium' in caplog.text
    assert 'fvn: fvn_pct <- [(col("fvn_share")) * (dyn int: 100)], %' in caplog.text


def test_the_log_remembers_only_recent_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past `_LOGGED_MOST` readings the oldest is forgotten, so a sweep of ever-different inputs holds no more."""
    monkeypatch.setattr(frames, "_LOGGED_MOST", 2)
    frames._LOGGED.clear()
    for description in ("a", "b", "c"):
        frames._log_once(description)
    assert list(frames._LOGGED) == ["b", "c"]


def test_values_no_100g_could_hold_warn() -> None:
    """Sodium in mg read as g, or energy past pure fat's, warns: those units are wrong."""
    frame = pl.DataFrame({**NAMED, "sodium": [400.0], "energy_kj": [5000.0], "fvn_points": [0]})
    with pytest.warns(ImpossibleInputWarning, match="energy_kj 1 above 3700 kJ, sodium 1 above 100 g"):
        score_frame(frame, "2004")


def test_lacking_inputs_raise_before_collecting() -> None:
    """A lazy frame without an input a scheme needs fails when the scoring is built, not when it runs."""
    frame = pl.LazyFrame({"energy_kj": [1500.0], "satfat": [4.5], "sugars": [20.0], "sodium": [0.5]})
    with pytest.raises(ValueError, match="needs protein or protein_points; fibre or fibre_points; fvn_pct or fvn_"):
        score_frame(frame, "2004")


def test_frames_of_points_score() -> None:
    """A frame of points already scored, such as a retailer's own, scores from them as named."""
    frame = pl.DataFrame(
        {
            "energy_points": [4, 10],
            "satfat_points": [4, 1],
            "sugars_points": [4, 0],
            "sodium_points": [5, 0],
            "protein_points": [5, 5],
            "fibre_points": [3, 0],
            "fvn_points": [0, 5],
        }
    )
    assert score_frame(frame, "2004")["npm_2004"].to_list() == [14, 1]


def test_empty_frames_score() -> None:
    """No rows, no scores, and the columns' types all the same."""
    frame = pl.DataFrame(schema=dict.fromkeys((*NAMED, "fvn_points"), pl.Float64))
    scored = score_frame(frame, "2004")
    assert scored.height == 0
    assert scored.schema["npm_2004"] == pl.Int64


def test_band_expr_bands_as_band_does(panels: pl.DataFrame) -> None:
    """The expression is the numpy banding, null where a value is."""
    got = panels.select(band_expr("2018", "free_sugars", "free_sug_100g")).to_series().to_numpy()
    np.testing.assert_array_equal(got, band("2018", "free_sugars", panels["free_sug_100g"].to_numpy()))
    nulls = pl.DataFrame({"x": [None, 0.0]}, schema={"x": pl.Float64})
    assert nulls.select(band_expr("2018", "free_sugars", "x")).to_series().to_list() == [None, 0]
