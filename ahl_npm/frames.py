"""NPM scores on polars frames and expressions.

The scoring reuses the numpy implementation `ahl_npm.scoring.score`, s.t. scoring a dataframe is consistent with
scoring an array of its rows, and scoring an expression is consistent with scoring a series of its values.
"""

import logging
import threading
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TypeVar

import polars as pl

from ahl_npm.schemes import Scheme, get_scheme
from ahl_npm.scoring import INPUTS, _check_inputs, _inputs_used, _score, band, describe_inputs, output_names

logger = logging.getLogger("ahl_npm")

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)

# Temporary column holding a frame's scoring struct before its fields are split into columns.
_SCORING = "__ahl_npm_scoring"

# Recently logged input descriptions, s.t. each is only logged once however often a scoring is rebuilt (e.g. in a
# sweep or a dashboard). Only the latest `_LOGGED_MOST` are kept, so memory use stays bounded when inputs vary.
_LOGGED: dict[str, None] = {}
_LOGGED_MOST = 256


def _log_once(description: str) -> None:
    if description in _LOGGED:
        return
    _LOGGED[description] = None
    if len(_LOGGED) > _LOGGED_MOST:
        del _LOGGED[next(iter(_LOGGED))]
    logger.info(description)


def _as_expr(column: str | pl.Expr) -> pl.Expr:
    return pl.col(column) if isinstance(column, str) else column


def _output_dtypes(scheme: str | int | Scheme) -> dict[str, pl.DataType]:
    """Return the polars dtype of each of `output_names`: integers for points and scores, and a boolean flag."""
    return {name: pl.Boolean if name == "protein_counted" else pl.Int64 for name in output_names(scheme)}


def score_expr(scheme: str | int | Scheme, columns: Mapping[str, str | pl.Expr], *, warn: bool = True) -> pl.Expr:
    """Return an expression for each row's NPM scoring, as a struct with a field per `output_names`, null if unknown.

    Inputs the scheme doesn't use are ignored, so the same mapping can be used for both schemes. The expression only
    warns once, with the counts from the first batch of rows with missing or implausible values, which is the whole
    frame unless it is scored in batches (e.g. by a streaming collect).

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        columns (Mapping[str, str | pl.Expr]): A column name or expression for each input, by `score`'s names.
        warn (bool): Whether to warn, with a summary, of missing inputs and implausible density values.

    Returns:
        pl.Expr: The scoring as a struct, e.g. to `.struct.unnest()` or `.struct.field()`.

    Raises:
        TypeError: If an input is not one of `INPUTS`.
        ValueError: If a component has none of the inputs it can be scored from.
    """
    scheme = get_scheme(scheme)
    _check_inputs(scheme, columns)
    used = _inputs_used(scheme)
    names = [name for name in INPUTS if name in columns and name in used]
    _log_once(describe_inputs(scheme, {name: columns[name] for name in names}))
    dtypes = _output_dtypes(scheme)
    warned = not warn
    claim = threading.Lock()

    def score_batch(rows: pl.Series) -> pl.Series:
        """Score a batch of rows (a struct of the inputs) with numpy."""
        nonlocal warned
        inputs = rows.struct.unnest()
        scored, problems = _score(scheme, {name: inputs[name].to_numpy() for name in names}, check=not warned)
        if problems:
            with claim:  # A streaming collect scores batches concurrently, so only the first batch with problems warns
                first, warned = not warned, True
            if first:
                for category, message in problems:
                    warnings.warn(message, category, stacklevel=2)
        fields = [pl.Series(name, values, nan_to_null=True).cast(dtypes[name]) for name, values in scored.items()]
        return pl.DataFrame(fields).to_struct(rows.name)

    struct = pl.struct([_as_expr(columns[name]).cast(pl.Float64).alias(name) for name in names])
    return struct.map_batches(score_batch, return_dtype=pl.Struct(dtypes), is_elementwise=True)


def score_frame(
    frame: FrameT,
    scheme: str | int | Scheme,
    columns: Mapping[str, str | pl.Expr] | None = None,
    *,
    prefix: str = "",
    suffix: str | None = None,
    outputs: Sequence[str] | Mapping[str, str] | None = None,
    warn: bool = True,
) -> FrameT:
    """Return `frame` with each row's NPM scoring under `scheme` added as columns.

    The columns added are `outputs` (by default all of `output_names`, i.e. each component's points, `a_points`,
    `c_points`, `protein_counted`, `npm` and `cnpm`), each named between `prefix` and `suffix`, e.g.
    `energy_points_2018`, ..., `cnpm_2018` by default.

    Args:
        frame (FrameT): The rows to score, as a `pl.DataFrame` or `pl.LazyFrame`.
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        columns (Mapping[str, str | pl.Expr] | None): A column name or expression for each input, by `score`'s names.
            If None, the frame's columns named as inputs are used.
        prefix (str): Prefix for each new column's name.
        suffix (str | None): Suffix for each new column's name, `_<scheme>` if None.
        outputs (Sequence[str] | Mapping[str, str] | None): Which of `output_names` to add: a single name, a sequence
            of names, or a mapping to new names (placed between `prefix` and `suffix`). None adds all of them.
        warn (bool): Whether to warn, with a summary, of missing inputs and of implausible density values.

    Returns:
        FrameT: `frame` with the new columns, of the same type.

    Raises:
        ValueError: If an output is not one of `output_names`, two outputs would have the same name, a new column
            would replace an input column, or the frame already has the temporary column the scoring uses.
    """
    scheme = get_scheme(scheme)
    suffix = f"_{scheme.name}" if suffix is None else suffix
    present = frame.collect_schema().names()
    if _SCORING in present:
        raise ValueError(f"The frame already has a column called {_SCORING}, which the scoring uses internally")
    if columns is None:
        columns = {name: name for name in INPUTS if name in present}
    if isinstance(outputs, str):
        outputs = (outputs,)
    known = output_names(scheme)
    chosen = (
        dict.fromkeys(known)
        if outputs is None
        else dict(outputs)
        if isinstance(outputs, Mapping)
        else dict.fromkeys(outputs)
    )
    unknown = sorted(set(chosen) - set(known))
    if unknown:
        raise ValueError(f"No NPM {scheme.name} outputs {unknown}: the outputs are {known}")
    named = {output: f"{prefix}{name or output}{suffix}" for output, name in chosen.items()}
    shared = sorted(name for name, count in Counter(named.values()).items() if count > 1)
    if shared:
        raise ValueError(f"Outputs would share the names {shared}")

    used = {column for value in columns.values() for column in _as_expr(value).meta.root_names()}
    clashes = sorted(used & set(named.values()))
    if clashes:
        raise ValueError(f"The scoring would replace input columns {clashes}: give a prefix, suffix or other outputs")

    scoring = pl.col(_SCORING)
    fields = [scoring.struct.field(output).alias(name) for output, name in named.items()]
    return (
        frame.with_columns(score_expr(scheme, columns, warn=warn).alias(_SCORING)).with_columns(fields).drop(_SCORING)
    )


def band_expr(scheme: str | int | Scheme, component: str, values: str | pl.Expr) -> pl.Expr:
    """Return an expression for the points one component gives each value per 100g, null where a value is missing.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        component (str): The component, e.g. "free_sugars", with values in its main input's units (e.g. kJ for energy).
        values (str | pl.Expr): The values per 100g, as a column name or an expression.

    Returns:
        pl.Expr: The points as integers.
    """
    scheme = get_scheme(scheme)
    scheme.component(component)  # Raise here, rather than at collect, if the scheme doesn't have the component

    def band_batch(batch: pl.Series) -> pl.Series:
        """Band a batch of values with numpy."""
        return pl.Series(batch.name, band(scheme, component, batch.to_numpy()), nan_to_null=True).cast(pl.Int64)

    return _as_expr(values).cast(pl.Float64).map_batches(band_batch, return_dtype=pl.Int64, is_elementwise=True)
