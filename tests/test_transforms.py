import numpy as np
import polars as pl
import pytest

from ahl_npm import from_cnpm, less_healthy, to_cnpm


def test_cnpm_is_70_minus_twice_the_score() -> None:
    """The cNPM score is 70 - 2 x NPM, only clipped when asked, for numbers, arrays and polars expressions alike."""
    assert to_cnpm(40) == -10
    assert to_cnpm(-20) == 110
    assert to_cnpm(np.array([40, -20]), clip=True).tolist() == [0, 100]
    assert pl.select(to_cnpm(pl.lit(-20), clip=True)).item() == 100
    assert from_cnpm(to_cnpm(np.arange(-20, 41))).tolist() == list(range(-20, 41))


def test_cnpm_changes_take_no_offset() -> None:
    """A change in score of +3 is a change in cNPM of -6 (and back again), and only scores can be clipped."""
    assert to_cnpm(3, change=True) == -6
    assert from_cnpm(-6, change=True) == 3
    assert to_cnpm(5, change=True) == to_cnpm(5) - to_cnpm(0)
    with pytest.raises(ValueError, match="clip"):
        to_cnpm(3, clip=True, change=True)


def test_less_healthy_from_4_for_food_and_1_for_drink() -> None:
    """A food is less healthy from a score of 4 and a drink from 1, with NaN or null where the score is unknown."""
    npm = [3, 4, 0, 1, np.nan]
    drink = [False, False, True, True, False]
    np.testing.assert_array_equal(less_healthy("2018", npm, drink), [0, 1, 0, 1, np.nan])
    frame = pl.DataFrame({"npm": [3, 4, 0, 1, None], "drink": [False, False, True, None, False]})
    assert frame.select(less_healthy("2018", pl.col("npm"), pl.col("drink"))).to_series().to_list() == [
        False,
        True,
        False,
        None,
        None,
    ]


def test_less_healthy_needs_to_know_every_drink() -> None:
    """Whether each product is a drink must be given as booleans: a missing flag or a number raises an error."""
    got = less_healthy(2004, pl.Series([0, 5, None]), pl.Series([True, False, False]))
    np.testing.assert_array_equal(got, [0, 1, np.nan])
    for drink in ([True, None], [1.0, np.nan], pl.Series([True, None]), [1, 0]):
        with pytest.raises(TypeError, match="drink must be True or False"):
            less_healthy(2004, [5, 5], drink)
