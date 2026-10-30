"""Transformations of NPM scores: to and from cNPM, and whether a food or drink is less healthy."""

from typing import TypeVar

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from ahl_npm.constants import CNPM_BOUNDS, CNPM_OFFSET, CNPM_SCALE
from ahl_npm.schemes import Scheme, get_scheme

T = TypeVar("T")


def to_cnpm(npm: T, clip: bool = False, change: bool = False) -> T:
    """Convert NPM scores to cNPM, 70 - 2 x the score, s.t. higher is healthier; or changes in scores, -2 x the change.

    cNPM runs from -10 to 100 under NPM 2004/05, and up to 110 under NPM 2018 as its C points can reach 20.

    Args:
        npm (T): Scores, or changes in scores, as a number, numpy array, or polars expression or series.
        clip (bool): Whether to clip cNPM to between 0 and 100 (scores only).
        change (bool): Whether `npm` holds changes in scores, which don't take the offset.

    Returns:
        T: The cNPM scores or changes, of the same type as `npm`.

    Raises:
        ValueError: If asked to clip changes.
    """
    if change:
        if clip:
            raise ValueError("Only cNPM scores can be clipped, not changes in them")
        return CNPM_SCALE * npm
    cnpm = CNPM_OFFSET + CNPM_SCALE * npm
    if not clip:
        return cnpm
    return cnpm.clip(*CNPM_BOUNDS) if hasattr(cnpm, "clip") else min(max(cnpm, CNPM_BOUNDS[0]), CNPM_BOUNDS[1])


def from_cnpm(cnpm: T, change: bool = False) -> T:
    """Convert cNPM scores back to NPM scores, (70 - cNPM) / 2; or changes in cNPM, -cNPM / 2.

    Args:
        cnpm (T): cNPM scores, or changes in them, as a number, numpy array, or polars expression or series.
        change (bool): Whether `cnpm` holds changes in scores, which don't take the offset.

    Returns:
        T: The NPM scores or changes, of the same type as `cnpm`.
    """
    return cnpm / CNPM_SCALE if change else (cnpm - CNPM_OFFSET) / CNPM_SCALE


def less_healthy(
    scheme: str | int | Scheme, npm: ArrayLike | pl.Expr, drink: ArrayLike | pl.Expr
) -> np.ndarray | pl.Expr:
    """Return whether each product is less healthy under `scheme`, i.e. a food scoring 4 or more, or a drink 1 or more.

    Whether a product is in scope of HFSS legislation is a separate question: a product is only HFSS if it is both in
    scope and less healthy. Whether each product is a drink must be given, as it is never guessed.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        npm (ArrayLike | pl.Expr): The scores, as numbers, arrays or polars series, or a polars expression.
        drink (ArrayLike | pl.Expr): Whether each product is a drink, as booleans with none missing, or a polars
            expression.

    Returns:
        np.ndarray | pl.Expr: Whether each product is less healthy: an expression if either input is one (null where
            either input is), otherwise floats (1 or 0, NaN where the score is missing).

    Raises:
        TypeError: If `drink` is not an expression and isn't all booleans, e.g. because one is missing.
    """
    scheme = get_scheme(scheme)
    food_cut, drink_cut = scheme.less_healthy_food, scheme.less_healthy_drink
    if isinstance(npm, pl.Expr) or isinstance(drink, pl.Expr):
        npm, drink = (x if isinstance(x, pl.Expr) else pl.lit(x) for x in (npm, drink))
        return npm >= drink.cast(pl.Int64) * (drink_cut - food_cut) + food_cut
    npm, drink = np.asarray(npm, dtype=float), np.asarray(drink)
    if drink.dtype != bool:
        raise TypeError(f"drink must be True or False for every product, not {drink.dtype} values")
    return np.where(np.isnan(npm), np.nan, npm >= np.where(drink, drink_cut, food_cut))
