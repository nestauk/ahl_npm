"""NPM scores on polars frames and expressions, under whatever names a frame gives its columns.

The scoring is `ahl_npm.scoring.score`, run on each batch of rows, so a frame scores exactly as arrays do, eager, lazy
or streaming. Inputs are taken by `score`'s names (`energy_kcal`, `salt`, ...): `columns` maps them to a frame's columns
or to expressions, and without it a frame's columns already named as inputs are used. Each scoring logs (at INFO, to
the `ahl_npm` logger, once a process for each different reading) what it reads for each component and in what units
(`describe_inputs`).
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
from ahl_npm.transforms import to_cnpm

logger = logging.getLogger("ahl_npm")

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)

# The temporary column a frame's scoring sits in before its fields are spread into columns.
_SCORING = "__ahl_npm_scoring"

# The input descriptions logged lately, each logged once however often a scoring is rebuilt (in a sweep, a dashboard),
# the oldest forgotten past `_LOGGED_MOST` so memory stays flat when inputs vary from call to call.
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
    """Return each of `output_names` with its polars type: points and scores as integers, and a flag."""
    return {name: pl.Boolean if name == "protein_counted" else pl.Int64 for name in output_names(scheme)}


def score_expr(scheme: str | int | Scheme, columns: Mapping[str, str | pl.Expr], *, warn: bool = True) -> pl.Expr:
    """Return an expression for each row's NPM scoring, as a struct of `output_names`' fields, null where unknown.

    Inputs the scheme doesn't read are left out, so one mapping serves both schemes. The expression warns once, with the
    counts of the first batch of rows it finds anything in: the whole frame, unless it is scored in batches, as a
    streaming collect does.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        columns (Mapping[str, str | pl.Expr]): Each input `score` takes, by its name, as a column or an expression.
        warn (bool): Whether to warn, with a summary, of missing inputs and of values no 100 g could hold.

    Returns:
        pl.Expr: The scoring, a struct to `.struct.unnest()` or `.struct.field()`.

    Raises:
        TypeError: If an input is not one of `INPUTS`.
        ValueError: If a component has none of the inputs it scores from.
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
        """Score a batch of rows, a struct of the inputs, through numpy."""
        nonlocal warned
        inputs = rows.struct.unnest()
        scored, problems = _score(scheme, {name: inputs[name].to_numpy() for name in names}, check=not warned)
        if problems:
            with claim:  # A streaming collect scores batches at once: only the first to find problems warns
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
    cnpm: bool = False,
    warn: bool = True,
) -> FrameT:
    """Return `frame` with each row's NPM scoring under `scheme` added as columns.

    The columns are `outputs`, by default all of `output_names` (each component's points, `a_points`, `c_points`,
    `protein_counted` and `npm`), and `cnpm` if asked for, each between `prefix` and `suffix`: `energy_points_2018`,
    ..., `npm_2018` by default.

    Args:
        frame (FrameT): The rows to score, a `pl.DataFrame` or `pl.LazyFrame`.
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        columns (Mapping[str, str | pl.Expr] | None): Inputs by `score`'s names, as columns or expressions. If None, the
            frame's columns named as inputs.
        prefix (str): Put before each new column's name.
        suffix (str | None): Put after each new column's name, `_<scheme>` if None.
        outputs (Sequence[str] | Mapping[str, str] | None): Which of `output_names`, and "cnpm" (70 - 2 x the score),
            to add: one, a sequence of them, or a mapping of them to the names to give them (between `prefix` and
            `suffix`). None adds all of `output_names`.
        cnpm (bool): Whether to add "cnpm" to `outputs` too.
        warn (bool): Whether to warn, with a summary, of missing inputs and of values no 100 g could hold.

    Returns:
        FrameT: `frame` with the scoring's columns, of the same type.

    Raises:
        ValueError: If an output is not one of `output_names` or "cnpm", two would share a name, a new column would
            replace an input column, or the frame has the column the scoring needs for itself.
    """
    scheme = get_scheme(scheme)
    suffix = f"_{scheme.name}" if suffix is None else suffix
    present = frame.collect_schema().names()
    if _SCORING in present:
        raise ValueError(f"The frame has a column {_SCORING}, which the scoring needs for itself")
    if columns is None:
        columns = {name: name for name in INPUTS if name in present}
    if isinstance(outputs, str):
        outputs = (outputs,)
    known = (*output_names(scheme), "cnpm")
    chosen = (
        dict.fromkeys(output_names(scheme))
        if outputs is None
        else dict(outputs)
        if isinstance(outputs, Mapping)
        else dict.fromkeys(outputs)
    )
    if cnpm:
        chosen.setdefault("cnpm", None)
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
    fields = [
        (to_cnpm(scoring.struct.field("npm")) if output == "cnpm" else scoring.struct.field(output)).alias(name)
        for output, name in named.items()
    ]
    return (
        frame.with_columns(score_expr(scheme, columns, warn=warn).alias(_SCORING)).with_columns(fields).drop(_SCORING)
    )


def band_expr(scheme: str | int | Scheme, component: str, values: str | pl.Expr) -> pl.Expr:
    """Return an expression for the points one component gives values per 100 g, null where a value is.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        component (str): The component, such as "free_sugars", in the units its first input takes (energy in kJ).
        values (str | pl.Expr): The values per 100 g, as a column or an expression.

    Returns:
        pl.Expr: The points, as integers.
    """
    scheme = get_scheme(scheme)
    scheme.component(component)  # Raise here, not at collect, for a component the scheme lacks.

    def band_batch(batch: pl.Series) -> pl.Series:
        """Band a batch of values through numpy."""
        return pl.Series(batch.name, band(scheme, component, batch.to_numpy()), nan_to_null=True).cast(pl.Int64)

    return _as_expr(values).cast(pl.Float64).map_batches(band_batch, return_dtype=pl.Int64, is_elementwise=True)
