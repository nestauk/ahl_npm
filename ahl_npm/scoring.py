"""NPM scoring with numpy, the single implementation of the scoring logic, which the polars functions also use.

Inputs are densities per 100g as sold: energy in kJ (or kcal), the fruit, vegetable and nut share (NPM 2004/05) and the
fruit, vegetable, nut and seed share (NPM 2018) in %, and everything else in g/100g. Any component can instead be given
as points that have already been scored, e.g. from a category lookup. Inputs can be numbers or arrays, which broadcast
together as in numpy, e.g. to score one product against multiple possible free sugars values. Inputs a scheme doesn't
use are ignored, so the same inputs can be passed to both schemes.

Missing values are NaN (polars nulls are converted to NaN). A missing value scores NaN points, so any score it feeds
into is NaN too, and a warning summarises what was missing. Implausible density values, i.e. below 0 or above the
maximum in `_UNITS`, are likely to be in the wrong units and are also warned about.
"""

import warnings
from collections.abc import Callable, Collection, Mapping
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ahl_npm.constants import KJ_PER_KCAL, SALT_PER_SODIUM
from ahl_npm.schemes import COMPONENTS, Component, Scheme, get_scheme
from ahl_npm.transforms import to_cnpm

# Whether each component adds to the score (A) or is subtracted from it (C).
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

# The inputs each component can be scored from, in order of preference: the quantity the scheme scores, then a
# converted stand-in. A component not listed here is scored from the input with its own name.
_Convert = Callable[[np.ndarray], np.ndarray] | Literal["points"] | None
_VALUE_SOURCES: dict[str, tuple[tuple[str, _Convert], ...]] = {
    "energy": (("energy_kj", None), ("energy_kcal", lambda kcal: kcal * KJ_PER_KCAL)),
    "sodium": (("sodium", None), ("salt", lambda salt: salt / SALT_PER_SODIUM)),
    "salt": (("salt", None), ("sodium", lambda sodium: sodium * SALT_PER_SODIUM)),
    "fvn": (("fvn_pct", None),),
    "fvns": (("fvns_pct", None),),
}
# How each stand-in is converted, by component and input, as shown by `describe_inputs`.
_CONVERSIONS: dict[tuple[str, str], str] = {
    ("energy", "energy_kcal"): f"x {KJ_PER_KCAL:g} to kJ",
    ("sodium", "salt"): f"/ {SALT_PER_SODIUM:g} to sodium",
    ("salt", "sodium"): f"x {SALT_PER_SODIUM:g} to salt",
}

# Each input's unit and its maximum plausible value per 100g: the energy of pure fat (37 kJ or 9 kcal per gram) for
# energy, and 100g or 100% otherwise. Values above this, or below 0, are flagged as implausible.
_UNITS: dict[str, tuple[str, float]] = {
    "energy_kj": ("kJ", 3700.0),
    "energy_kcal": ("kcal", 900.0),
    **dict.fromkeys(("satfat", "sugars", "free_sugars", "sodium", "salt", "protein", "fibre"), ("g", 100.0)),
    "fvn_pct": ("%", 100.0),
    "fvns_pct": ("%", 100.0),
}


def _sources_of(component: str) -> tuple[tuple[str, _Convert], ...]:
    """Return the inputs a component can be scored from, in order of preference: its values, then its points."""
    return (*_VALUE_SOURCES.get(component, ((component, None),)), (f"{component}_points", "points"))


_SOURCES: dict[str, tuple[tuple[str, _Convert], ...]] = {c: _sources_of(c) for c in COMPONENTS}


def _sources(component: str) -> tuple[tuple[str, _Convert], ...]:
    return _SOURCES[component]


# All inputs `score` accepts: the densities per 100g, then each component's points.
INPUTS: tuple[str, ...] = (
    *dict.fromkeys(name for c in COMPONENTS for name, how in _sources(c) if how != "points"),
    *(f"{c}_points" for c in COMPONENTS),
)

# Whether each input adds to the score (A) or is subtracted from it (C).
_INPUT_KINDS: dict[str, str] = {name: _KINDS[c] for c in COMPONENTS for name, _ in _sources(c)}


class MissingInputWarning(UserWarning):
    """Some inputs were missing, so some points and scores are missing too."""


class ImpossibleInputWarning(UserWarning):
    """Some inputs have implausible density values, e.g. from being in the wrong units."""


# The warnings a scoring would raise, as (category, message) pairs.
Problems = list[tuple[type[UserWarning], str]]


def output_names(scheme: str | int | Scheme) -> tuple[str, ...]:
    """Return the names of the outputs `score` returns for a scheme, in order.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.

    Returns:
        tuple[str, ...]: Each component's `<name>_points`, then `a_points`, `c_points`, `protein_counted`, `npm` and
            `cnpm`.
    """
    components = get_scheme(scheme).components
    return (*(f"{comp.name}_points" for comp in components), "a_points", "c_points", "protein_counted", "npm", "cnpm")


def _inputs_used(scheme: str | int | Scheme) -> frozenset[str]:
    """Return the inputs used by any of `scheme`'s components, by `score`'s names; `score` ignores any others."""
    return frozenset(name for comp in get_scheme(scheme).components for name, _ in _sources(comp.name))


def _check_inputs(scheme: str | int | Scheme, inputs: Collection[str]) -> None:
    """Raise a TypeError if an input isn't one of `INPUTS`, or a ValueError if a component of `scheme` has no inputs."""
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
    """Return a description of the inputs each component of `scheme` reads, and in what units, as `score_frame` logs.

    This makes the scoring's assumptions about each input explicit so that inputs in the wrong units (e.g. mg read as g,
    or a 0 to 1 share read as a %) can be spotted before the scores are used.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        inputs (Mapping[str, object]): The inputs given, by `score`'s names, mapped to their source, e.g. a column name,
            an expression or an array.

    Returns:
        str: One line per component, listing each input it reads in order of preference, with its source, unit and
            any conversion.
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
    """Return a summary of implausible density values by input, or None if there are none."""
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
    """Return the points one component gives each value per 100g, i.e. those for the highest threshold it is above.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        component (str): The component, e.g. "free_sugars", with values in its main input's units (e.g. kJ for energy).
        values (ArrayLike): The values per 100g.
        strict (bool): Whether to require every value, returning integers; otherwise floats, NaN where missing.
        warn (bool): Whether to warn of implausible density values.

    Returns:
        np.ndarray: The points.

    Raises:
        ValueError: If `strict` and a value is missing.
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if warn and (impossible := _impossible({_sources(component)[0][0]: values})):
        warnings.warn(f"Implausible {component} density values: {impossible}", ImpossibleInputWarning, stacklevel=2)
    points = get_scheme(scheme).component(component).band(values)
    if not strict:
        return points
    if np.isnan(points).any():
        raise ValueError(f"{int(np.isnan(points).sum()):,} {component} values are missing, which strict does not allow")
    return points.astype(int)


def _component_points(scheme: Scheme, component: Component, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return a component's points, scoring each row from the first of the component's inputs it has."""
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
    """Raise if the points given are not ones the component can score, e.g. if they are from another scheme."""
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
    """Calculate NPM scores plus each component's points, the A and C points, and whether protein counts.

    A component can be scored from more than one input, and each row uses the first it has: the quantity the scheme
    scores, then a stand-in converted to it (kcal x 4.184 for kJ, salt / 2.5 for sodium, sodium x 2.5 for salt), then
    the component's points if given. Protein only counts below 11 A points, or if fruit, vegetables and nuts (and
    seeds, for NPM 2018) score 5 points, so a row where protein doesn't count is scored without it, even if missing.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        energy_kj (ArrayLike | None): Energy in kJ per 100g.
        energy_kcal (ArrayLike | None): Energy in kcal per 100g.
        satfat (ArrayLike | None): Saturated fat in g per 100g.
        sugars (ArrayLike | None): Total sugars in g per 100g, scored by NPM 2004/05.
        free_sugars (ArrayLike | None): Free sugars in g per 100g, scored by NPM 2018.
        sodium (ArrayLike | None): Sodium in g per 100g, scored by NPM 2004/05.
        salt (ArrayLike | None): Salt in g per 100g, scored by NPM 2018.
        protein (ArrayLike | None): Protein in g per 100g.
        fibre (ArrayLike | None): Fibre (AOAC method) in g per 100g.
        fvn_pct (ArrayLike | None): Fruit, vegetables and nuts as a % of the product, scored by NPM 2004/05.
        fvns_pct (ArrayLike | None): Fruit, vegetables, nuts and seeds as a % of the product, scored by NPM 2018.
        energy_points (ArrayLike | None): Energy points, already scored under `scheme`.
        satfat_points (ArrayLike | None): Saturated fat points, already scored under `scheme`.
        sugars_points (ArrayLike | None): Total sugars points, already scored under NPM 2004/05.
        free_sugars_points (ArrayLike | None): Free sugars points, already scored under NPM 2018.
        sodium_points (ArrayLike | None): Sodium points, already scored under NPM 2004/05.
        salt_points (ArrayLike | None): Salt points, already scored under NPM 2018.
        protein_points (ArrayLike | None): Protein points, already scored under `scheme`.
        fibre_points (ArrayLike | None): Fibre points, already scored under `scheme`.
        fvn_points (ArrayLike | None): Fruit, vegetables and nuts points, already scored under NPM 2004/05, e.g. from
            a category lookup.
        fvns_points (ArrayLike | None): Fruit, vegetables, nuts and seeds points, already scored under NPM 2018.
        strict (bool): Whether to require every input, returning integers; otherwise floats, NaN where unknown.
        warn (bool): Whether to warn, with a summary, of missing inputs and implausible density values.

    Returns:
        dict[str, np.ndarray]: The outputs by `output_names`, each in the inputs' broadcast shape: each component's
            `<name>_points`, `a_points`, `c_points` (as counted), `protein_counted` (1 or 0), `npm` and `cnpm` (70 - 2 x
            `npm`).

    Raises:
        TypeError: If an input is not one of `INPUTS`.
        ValueError: If a component has none of the inputs it can be scored from, the points given aren't ones it can
            score, or `strict` and an input is missing.
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
    """Score as `score` does, but return any warnings (if `check`) rather than raising them, for the frame functions."""
    scheme = get_scheme(scheme)
    _check_inputs(scheme, given)
    used = _inputs_used(scheme)
    values = {name: np.atleast_1d(np.asarray(value, dtype=float)) for name, value in given.items() if name in used}
    problems: Problems = []
    # Check before broadcasting, s.t. a single value given for every row is only counted once
    if check and (impossible := _impossible(values)):
        problems.append((ImpossibleInputWarning, f"NPM {scheme.name}: implausible density values: {impossible}"))
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
        "cnpm": to_cnpm(npm),
    }

    if strict:
        if missing := _missing(scheme, points, npm):
            raise ValueError(f"{missing}, which strict does not allow")
        return {name: values.astype(int) for name, values in scored.items()}, problems
    if check and (missing := _missing(scheme, points, npm)):
        problems.append((MissingInputWarning, missing))
    return scored, problems


def _missing(scheme: Scheme, points: Mapping[str, np.ndarray], npm: np.ndarray) -> str | None:
    """Return a summary of how many scores are missing and how many rows each component is missing from, if any."""
    missing = {name: int(np.isnan(p).sum()) for name, p in points.items()}
    missing = {name: n for name, n in missing.items() if n}
    if not missing:
        return None
    unscored = int(np.isnan(npm).sum())
    by_component = ", ".join(f"{name} {n:,}" for name, n in missing.items())
    note = "; protein is only needed where it counts" if "protein" in missing else ""
    return (
        f"NPM {scheme.name}: {unscored:,} of {npm.size:,} scores ({unscored / npm.size:.1%}) are missing due to missing"
        f" inputs (rows missing each component: {by_component}){note}"
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
    """Return the scorings with the lowest and highest scores for inputs anywhere between `low` and `high`.

    The score never decreases as an A input increases or as a C input decreases (including through the protein cap),
    so the lowest score has the A inputs at `low` and the C inputs at `high`, and the highest score the opposite. This
    means only two scorings are needed, however many inputs are uncertain.

    Args:
        scheme (str | int | Scheme): 2004, 2018 or a `Scheme`.
        low (Mapping[str, ArrayLike]): The uncertain inputs at their lowest, by `score`'s names.
        high (Mapping[str, ArrayLike]): The same inputs at their highest.
        strict (bool): Whether to require every input, as in `score`.
        warn (bool): Whether to warn of missing inputs and implausible density values.
        **fixed (ArrayLike): The known inputs, by `score`'s names.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: `score`'s outputs at the lowest score, then the highest.

    Raises:
        ValueError: If `low` and `high` give different inputs, or an input is given in both these and `fixed`.
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
