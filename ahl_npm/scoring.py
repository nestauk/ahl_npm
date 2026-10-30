"""NPM scores with numpy: the one place the scoring logic lives, which the polars functions run on too.

Inputs are per 100 g as sold: energy in kJ (or kcal), the fruit, vegetable and nut share (NPM 2004/05) and the fruit,
vegetable, nut and seed share (NPM 2018) in %, and the rest in g. Any component can instead be given its points, already
scored, such as a retailer's own or a category lookup's. Inputs are numbers or arrays, which broadcast together as
numpy's do, so one product's panel can be scored against many draws of its free sugars; one a scheme doesn't read is
ignored, so one set of inputs serves both schemes.

A missing value is NaN (polars' nulls arrive as NaN): it scores NaN points, so the score it feeds is NaN too, and a
warning sums up what was missing. A value no 100 g could hold, below 0 or above `_UNITS`' most, can only be in the wrong
units or wrong, and is warned of too.
"""

import warnings
from collections.abc import Callable, Collection, Mapping
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ahl_npm.constants import KJ_PER_KCAL, SALT_PER_SODIUM
from ahl_npm.schemes import COMPONENTS, Component, Scheme, get_scheme

# Whether each component counts towards the score (A) or against it (C).
_KINDS: dict[str, str] = {
    "energy": "A",
    "satfat": "A",
    "sugars": "A",
    "free_sugars": "A",
    "sodium": "A",
    "salt": "A",
    "protein": "C",
    "fibre": "C",
    "fvn": "C",
    "fvns": "C",
}

# The values a component can score, in order: the quantity a scheme scores, then a stand-in for it, converted. A
# component not listed scores the input of its own name.
_Convert = Callable[[np.ndarray], np.ndarray] | Literal["points"] | None
_VALUE_SOURCES: dict[str, tuple[tuple[str, _Convert], ...]] = {
    "energy": (("energy_kj", None), ("energy_kcal", lambda kcal: kcal * KJ_PER_KCAL)),
    "sodium": (("sodium", None), ("salt", lambda salt: salt / SALT_PER_SODIUM)),
    "salt": (("salt", None), ("sodium", lambda sodium: sodium * SALT_PER_SODIUM)),
    "fvn": (("fvn_pct", None),),
    "fvns": (("fvns_pct", None),),
}
# How each stand-in converts, by its component and input, as `describe_inputs` says it.
_CONVERSIONS: dict[tuple[str, str], str] = {
    ("energy", "energy_kcal"): f"x {KJ_PER_KCAL:g} to kJ",
    ("sodium", "salt"): f"/ {SALT_PER_SODIUM:g} to sodium",
    ("salt", "sodium"): f"x {SALT_PER_SODIUM:g} to salt",
}

# Each value input's unit, and the most of it 100 g of anything holds: pure fat's energy (37 kJ, 9 kcal a gram), and
# otherwise all of it. A value above that, or below 0, is in the wrong units or wrong.
_UNITS: dict[str, tuple[str, float]] = {
    "energy_kj": ("kJ", 3700.0),
    "energy_kcal": ("kcal", 900.0),
    **dict.fromkeys(("satfat", "sugars", "free_sugars", "sodium", "salt", "protein", "fibre"), ("g", 100.0)),
    "fvn_pct": ("%", 100.0),
    "fvns_pct": ("%", 100.0),
}


def _sources_of(component: str) -> tuple[tuple[str, _Convert], ...]:
    """Return the inputs a component scores from in the order a row takes them: its values, then its points."""
    return (*_VALUE_SOURCES.get(component, ((component, None),)), (f"{component}_points", "points"))


_SOURCES: dict[str, tuple[tuple[str, _Convert], ...]] = {c: _sources_of(c) for c in COMPONENTS}


def _sources(component: str) -> tuple[tuple[str, _Convert], ...]:
    return _SOURCES[component]


# The inputs a score can take, all per 100 g: the values, then each component's points.
INPUTS: tuple[str, ...] = (
    *dict.fromkeys(name for c in COMPONENTS for name, how in _sources(c) if how != "points"),
    *(f"{c}_points" for c in COMPONENTS),
)

# Whether each input counts towards the score (A) or against it (C).
_INPUT_KINDS: dict[str, str] = {name: _KINDS[c] for c in COMPONENTS for name, _ in _sources(c)}


class MissingInputWarning(UserWarning):
    """Some inputs were missing, so some points and scores are too."""


class ImpossibleInputWarning(UserWarning):
    """Some inputs hold values no 100 g could, so are in the wrong units or wrong."""


# What a scoring would warn of: each warning's category and message.
Problems = list[tuple[type[UserWarning], str]]


def output_names(scheme: str | int | Scheme) -> tuple[str, ...]:
    """Return the names of what `score` returns for a scheme, in order.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.

    Returns:
        tuple[str, ...]: Each component's `<name>_points`, then `a_points`, `c_points`, `protein_counted` and `npm`.
    """
    components = get_scheme(scheme).components
    return (*(f"{comp.name}_points" for comp in components), "a_points", "c_points", "protein_counted", "npm")


def _inputs_used(scheme: str | int | Scheme) -> frozenset[str]:
    """Return the inputs some component of `scheme` reads, by `score`'s names: `score` ignores any other."""
    return frozenset(name for comp in get_scheme(scheme).components for name, _ in _sources(comp.name))


def _check_inputs(scheme: str | int | Scheme, inputs: Collection[str]) -> None:
    """Raise if an input is not one of `INPUTS` (TypeError), or a component of `scheme` has none it scores from."""
    scheme = get_scheme(scheme)
    unknown = sorted(set(inputs) - set(INPUTS))
    if unknown:
        raise TypeError(f"Unknown NPM inputs {unknown}: the inputs are {INPUTS}")
    lacking = [
        " or ".join(name for name, _ in _sources(comp.name))
        for comp in scheme.components
        if not any(name in inputs for name, _ in _sources(comp.name))
    ]
    if lacking:
        raise ValueError(f"NPM {scheme.name} needs {'; '.join(lacking)}")


def _describe(source: object) -> str:
    if isinstance(source, str):
        return f'"{source}"'
    if isinstance(source, np.ndarray):
        return f"an array of {source.size:,}"
    return str(source)


def describe_inputs(scheme: str | int | Scheme, inputs: Mapping[str, object]) -> str:
    """Return what each component of `scheme` reads from `inputs`, in what units, as `score_frame` logs it.

    It says what the scoring assumes of each input, so a column in the wrong units (mg read as g, a 0 to 1 share read as
    a %) shows before its scores are trusted.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        inputs (Mapping[str, object]): The inputs given, by `score`'s names, to what each is: a column's name, an
            expression or an array.

    Returns:
        str: A line per component: each input it reads in the order a row takes them, with its source, unit and any
            conversion.
    """
    scheme = get_scheme(scheme)
    lines = [f"NPM {scheme.name} reads, per 100 g:"]
    for comp in scheme.components:
        reads = [
            f"{name} <- {_describe(inputs[name])}, "
            + ("points as given" if convert == "points" else _UNITS[name][0])
            + (f" {_CONVERSIONS[comp.name, name]}" if (comp.name, name) in _CONVERSIONS else "")
            for name, convert in _sources(comp.name)
            if name in inputs
        ]
        lines.append(f"  {comp.name}: {'; else '.join(reads) if reads else 'nothing'}")
    return "\n".join(lines)


def _impossible(inputs: Mapping[str, np.ndarray]) -> str | None:
    """Return a summary of the values no 100 g could hold, by input, or None if there are none."""
    found = []
    for name, values in inputs.items():
        if name not in _UNITS:
            continue
        unit, most = _UNITS[name]
        below, above = int((values < 0).sum()), int((values > most).sum())
        if below:
            found.append(f"{name} {below:,} below 0")
        if above:
            found.append(f"{name} {above:,} above {most:g} {unit}")
    return ", ".join(found) or None


def band(
    scheme: str | int | Scheme, component: str, values: ArrayLike, *, strict: bool = False, warn: bool = True
) -> np.ndarray:
    """Return the points one component gives values per 100 g: those for the highest threshold each is above.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        component (str): The component, such as "free_sugars", in the units its first input takes (energy in kJ).
        values (ArrayLike): The values per 100 g.
        strict (bool): Whether every value must be known, to return integers; otherwise floats, NaN where a value is.
        warn (bool): Whether to warn of values no 100 g could hold.

    Returns:
        np.ndarray: The points.

    Raises:
        ValueError: If `strict` and a value is missing.
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if warn and (impossible := _impossible({_sources(component)[0][0]: values})):
        warnings.warn(f"{component} values no 100 g could hold: {impossible}", ImpossibleInputWarning, stacklevel=2)
    points = get_scheme(scheme).component(component).band(values)
    if not strict:
        return points
    if np.isnan(points).any():
        raise ValueError(f"{int(np.isnan(points).sum()):,} {component} values are missing, which strict refuses")
    return points.astype(int)


def _component_points(scheme: Scheme, component: Component, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return a component's points, each row's from the first of the component's inputs it has."""
    points = None
    for name, convert in _sources(component.name):
        if name not in inputs:
            continue
        if convert == "points":
            these = inputs[name]
            _check_points(scheme, component, name, these)
        else:
            these = component.band(inputs[name] if convert is None else convert(inputs[name]))
        points = these if points is None else np.where(np.isnan(points), these, points)
    return points


def _check_points(scheme: Scheme, component: Component, name: str, points: np.ndarray) -> None:
    """Raise if points given are not ones the component can score, as from another scheme's scale."""
    wrong = ~(np.isin(points, component.point_values) | np.isnan(points))
    if wrong.any():
        raise ValueError(
            f"{name} must be NPM {scheme.name} {component.name} points, one of {component.point_values.astype(int)}:"
            f" {int(wrong.sum()):,} are not, such as {np.unique(points[wrong])[:5]}"
        )


def score(
    scheme: str | int | Scheme,
    *,
    energy_kj: ArrayLike | None = None,
    energy_kcal: ArrayLike | None = None,
    satfat: ArrayLike | None = None,
    sugars: ArrayLike | None = None,
    free_sugars: ArrayLike | None = None,
    sodium: ArrayLike | None = None,
    salt: ArrayLike | None = None,
    protein: ArrayLike | None = None,
    fibre: ArrayLike | None = None,
    fvn_pct: ArrayLike | None = None,
    fvns_pct: ArrayLike | None = None,
    energy_points: ArrayLike | None = None,
    satfat_points: ArrayLike | None = None,
    sugars_points: ArrayLike | None = None,
    free_sugars_points: ArrayLike | None = None,
    sodium_points: ArrayLike | None = None,
    salt_points: ArrayLike | None = None,
    protein_points: ArrayLike | None = None,
    fibre_points: ArrayLike | None = None,
    fvn_points: ArrayLike | None = None,
    fvns_points: ArrayLike | None = None,
    strict: bool = False,
    warn: bool = True,
) -> dict[str, np.ndarray]:
    """Score NPM: each component's points, the A and C points, whether protein counts, and the score.

    A component can score from more than one input, and each row takes the first it has: the quantity the scheme
    scores, then a stand-in converted to it (kcal x 4.184 for kJ, salt / 2.5 for sodium, sodium x 2.5 for salt), then
    the component's points as given. Protein counts below 11 A points, or from there with 5 fruit, vegetable and nut
    points (and seed, for NPM 2018), so a row whose protein can't count scores without it, even with protein missing.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        energy_kj (ArrayLike | None): Energy, kJ per 100 g.
        energy_kcal (ArrayLike | None): Energy, kcal per 100 g.
        satfat (ArrayLike | None): Saturated fat, g per 100 g.
        sugars (ArrayLike | None): Total sugars, g per 100 g, which NPM 2004/05 scores.
        free_sugars (ArrayLike | None): Free sugars, g per 100 g, which NPM 2018 scores.
        sodium (ArrayLike | None): Sodium, g per 100 g, which NPM 2004/05 scores.
        salt (ArrayLike | None): Salt, g per 100 g, which NPM 2018 scores.
        protein (ArrayLike | None): Protein, g per 100 g.
        fibre (ArrayLike | None): Fibre by the AOAC method, g per 100 g.
        fvn_pct (ArrayLike | None): Fruit, vegetables and nuts as a % of the product, which NPM 2004/05 scores.
        fvns_pct (ArrayLike | None): Fruit, vegetables, nuts and seeds as a % of the product, which NPM 2018 scores.
        energy_points (ArrayLike | None): Energy's points under `scheme`, already scored.
        satfat_points (ArrayLike | None): Saturated fat's points under `scheme`, already scored.
        sugars_points (ArrayLike | None): Total sugars' NPM 2004/05 points, already scored.
        free_sugars_points (ArrayLike | None): Free sugars' NPM 2018 points, already scored.
        sodium_points (ArrayLike | None): Sodium's NPM 2004/05 points, already scored.
        salt_points (ArrayLike | None): Salt's NPM 2018 points, already scored.
        protein_points (ArrayLike | None): Protein's points under `scheme`, already scored.
        fibre_points (ArrayLike | None): Fibre's points under `scheme`, already scored.
        fvn_points (ArrayLike | None): Fruit, vegetables and nuts' NPM 2004/05 points, already scored, such as by
            Kantar's category lookup.
        fvns_points (ArrayLike | None): Fruit, vegetables, nuts and seeds' NPM 2018 points, already scored.
        strict (bool): Whether every input must be known, to return integers; otherwise floats, NaN where unknown.
        warn (bool): Whether to warn, with a summary, of missing inputs and of values no 100 g could hold.

    Returns:
        dict[str, np.ndarray]: By `output_names`, each in the inputs' broadcast shape: each component's `<name>_points`,
            `a_points`, `c_points` (as counted), `protein_counted` (1 or 0) and `npm`.

    Raises:
        TypeError: If an input is not one of `INPUTS`.
        ValueError: If a component has none of the inputs it scores from, points given aren't ones it can score, or
            `strict` and an input is missing.
    """
    arguments = locals()
    given = {name: arguments[name] for name in INPUTS if arguments[name] is not None}
    scored, problems = _score(scheme, given, strict=strict, check=warn)
    for category, message in problems:
        warnings.warn(message, category, stacklevel=2)
    return scored


def _score(
    scheme: str | int | Scheme, given: Mapping[str, ArrayLike], *, strict: bool = False, check: bool = True
) -> tuple[dict[str, np.ndarray], Problems]:
    """Score as `score` does, returning what it would warn of (if `check`) rather than warning, as frames need."""
    scheme = get_scheme(scheme)
    _check_inputs(scheme, given)
    used = _inputs_used(scheme)
    values = {name: np.atleast_1d(np.asarray(value, dtype=float)) for name, value in given.items() if name in used}
    problems: Problems = []
    # Checked before broadcasting, so a value given once counts once
    if check and (impossible := _impossible(values)):
        problems.append((ImpossibleInputWarning, f"NPM {scheme.name}: values no 100 g could hold: {impossible}"))
    inputs = dict(zip(values, np.broadcast_arrays(*values.values()), strict=True))

    points = {comp.name: _component_points(scheme, comp, inputs) for comp in scheme.components}
    a = sum(points[comp.name] for comp in scheme.components if comp.kind == "A")
    capped, unlocks = a >= scheme.protein_cap_a_points, points[scheme.protein_cap_component]
    counted = np.where(capped, (unlocks >= scheme.protein_cap_points).astype(float), 1.0)
    counted = np.where(np.isnan(a) | (capped & np.isnan(unlocks)), np.nan, counted)
    c = sum(points[comp.name] for comp in scheme.components if comp.kind == "C" and comp.name != "protein")
    c = np.where(np.isnan(counted), np.nan, c + np.where(counted == 1, points["protein"], 0.0))
    npm = a - c
    scored = {
        **{f"{name}_points": p for name, p in points.items()},
        "a_points": a,
        "c_points": c,
        "protein_counted": counted,
        "npm": npm,
    }

    if strict:
        if missing := _missing(scheme, points, npm):
            raise ValueError(f"{missing}, which strict refuses")
        return {name: values.astype(int) for name, values in scored.items()}, problems
    if check and (missing := _missing(scheme, points, npm)):
        problems.append((MissingInputWarning, missing))
    return scored, problems


def _missing(scheme: Scheme, points: Mapping[str, np.ndarray], npm: np.ndarray) -> str | None:
    """Return how many scores are missing and how many rows each component's inputs are missing from, if any are."""
    missing = {name: int(np.isnan(p).sum()) for name, p in points.items()}
    missing = {name: n for name, n in missing.items() if n}
    if not missing:
        return None
    unscored = int(np.isnan(npm).sum())
    by_component = ", ".join(f"{name} {n:,}" for name, n in missing.items())
    note = "; protein is only needed where it counts" if "protein" in missing else ""
    return (
        f"NPM {scheme.name}: {unscored:,} of {npm.size:,} scores ({unscored / npm.size:.1%}) are missing, for want of"
        f" inputs (rows missing each: {by_component}){note}"
    )


def score_range(
    scheme: str | int | Scheme,
    low: Mapping[str, ArrayLike],
    high: Mapping[str, ArrayLike],
    *,
    strict: bool = False,
    warn: bool = True,
    **fixed: ArrayLike,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the scorings with the lowest and the highest score for inputs anywhere from `low` to `high`.

    A score never falls as an A input rises or as a C input falls, protein's cap included, so the lowest comes with the
    A inputs at `low` and the C inputs at `high`, and the highest the other way round: two scorings, however many
    inputs are uncertain.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        low (Mapping[str, ArrayLike]): The uncertain inputs at their lowest, by `score`'s names.
        high (Mapping[str, ArrayLike]): The same inputs at their highest.
        strict (bool): Whether every input must be known, as `score`'s.
        warn (bool): Whether to warn of missing inputs and of values no 100 g could hold.
        **fixed (ArrayLike): The inputs that are known, by `score`'s names.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: `score`'s output at the lowest score, then the highest.

    Raises:
        ValueError: If `low` and `high` name different inputs, or name one `fixed` does too.
    """
    if set(low) != set(high):
        raise ValueError(f"low and high must name the same inputs, not {sorted(low)} and {sorted(high)}")
    if set(low) & set(fixed):
        raise ValueError(f"{sorted(set(low) & set(fixed))} are given both as uncertain and as fixed")
    _check_inputs(scheme, {*low, *fixed})
    at_lowest = {name: (low if _INPUT_KINDS[name] == "A" else high)[name] for name in low}
    at_highest = {name: (high if _INPUT_KINDS[name] == "A" else low)[name] for name in low}
    return (
        score(scheme, **fixed, **at_lowest, strict=strict, warn=warn),
        score(scheme, **fixed, **at_highest, strict=strict, warn=False),
    )
