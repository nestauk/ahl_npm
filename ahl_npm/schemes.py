"""NPM schemes: each component's thresholds and points, the protein cap and the less healthy cut-offs.

Each scheme is defined in a YAML file in `ahl_npm/schemes`, which is parsed once into a `Scheme`. `load_scheme` parses a
custom scheme from a YAML file with the same layout.
"""

from dataclasses import dataclass
from functools import cache, cached_property
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from ahl_npm.constants import BAND_ATOL, BAND_RTOL

SCHEMES: tuple[str, ...] = ("2004", "2018")

# The components a scheme can use, i.e. those `ahl_npm.scoring` knows the inputs for. `fvn` is NPM 2004/05's fruit,
# vegetables and nuts, and `fvns` is NPM 2018's fruit, vegetables, nuts and seeds.
COMPONENTS: tuple[str, ...] = (
    "energy",
    "satfat",
    "sugars",
    "free_sugars",
    "sodium",
    "salt",
    "protein",
    "fibre",
    "fvn",
    "fvns",
)

# Divisors converting a scheme's threshold units into the units inputs are given in (kJ, g or % per 100g).
_UNIT_DIVISORS: dict[str, int] = {"kJ": 1, "g": 1, "mg": 1000, "%": 1}

# Below this many values, a binary search is faster for banding than comparing against each threshold.
_SEARCH_BELOW: int = 512


@dataclass(frozen=True)
class Component:
    """A component of a scheme's score, and the points it gives a value per 100g.

    Attributes:
        name (str): The component, e.g. "energy" or "free_sugars".
        kind (str): "A" if the component adds to the score, "C" if it is subtracted from it.
        thresholds (tuple[float, ...]): The thresholds a value must be above for each point, in kJ, g or % per 100g.
        points (tuple[int, ...]): The points for being above each threshold.
    """

    name: str
    kind: Literal["A", "C"]
    thresholds: tuple[float, ...]
    points: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check the thresholds and points are valid, i.e. both increasing and of equal length."""
        if self.kind not in ("A", "C"):
            raise ValueError(f"{self.name}: kind must be 'A' or 'C', not {self.kind!r}")
        if len(self.points) != len(self.thresholds):
            raise ValueError(f"{self.name}: {len(self.points)} points for {len(self.thresholds)} thresholds")
        if any(b <= a for a, b in pairwise(self.thresholds)):
            raise ValueError(f"{self.name}: thresholds must increase")
        if any(b <= a for a, b in pairwise((0, *self.points))):
            raise ValueError(f"{self.name}: points must increase from 0")

    @cached_property
    def edges(self) -> np.ndarray:
        """The thresholds as values are compared against them, each raised by the tolerance in `ahl_npm.constants`."""
        thresholds = np.asarray(self.thresholds, dtype=float)
        return thresholds + BAND_ATOL + BAND_RTOL * np.abs(thresholds)

    @cached_property
    def point_values(self) -> np.ndarray:
        """Every value the component can score: the points for being above none, one, two, ... of the thresholds."""
        return np.asarray((0, *self.points), dtype=float)

    @cached_property
    def _band_edges(self) -> np.ndarray:
        # The edges with infinity appended, s.t. every number is at or below the last edge and NaN sorts after it.
        return np.append(self.edges, np.inf)

    @cached_property
    def _band_points(self) -> np.ndarray:
        # The points for being above none, one, two, ... of the edges, with NaN appended for missing values.
        return np.append(self.point_values, np.nan)

    def band(self, values: np.ndarray) -> np.ndarray:
        """Return the points for each value per 100g, i.e. those for the highest threshold it is above.

        Args:
            values (np.ndarray): Values per 100g as floats, NaN where missing.

        Returns:
            np.ndarray: The points as floats, NaN where a value is missing.
        """
        if values.size < _SEARCH_BELOW:
            return self._band_points[np.searchsorted(self._band_edges, values, side="left")]
        # For many values, comparing against each edge and counting is faster than a binary search.
        above = np.zeros(values.shape, dtype=np.uint8)
        for edge in self.edges:
            np.add(above, values > edge, out=above, casting="unsafe")
        above[np.isnan(values)] = len(self.edges) + 1
        return self._band_points[above]

    @property
    def max_points(self) -> int:
        """The maximum points the component can give."""
        return self.points[-1]


@dataclass(frozen=True)
class Scheme:
    """An NPM scheme: its components, the protein cap, and the less healthy cut-offs for food and drink.

    Attributes:
        name (str): The scheme, "2004" or "2018".
        source (str): The URL of the scheme's technical guidance.
        components (tuple[Component, ...]): The scheme's components, A then C.
        protein_cap_a_points (int): The A points at which protein stops counting, ...
        protein_cap_component (str): ... unless this component (`fvn` or `fvns`) scores ...
        protein_cap_points (int): ... at least this many points.
        less_healthy_food (int): The score at or above which a food is less healthy.
        less_healthy_drink (int): The score at or above which a drink is less healthy.
    """

    name: str
    source: str
    components: tuple[Component, ...]
    protein_cap_a_points: int
    protein_cap_component: str
    protein_cap_points: int
    less_healthy_food: int
    less_healthy_drink: int

    def __post_init__(self) -> None:
        """Check the scheme's components are all known, and include protein and the protein cap's component."""
        names = [c.name for c in self.components]
        unknown = sorted(set(names) - set(COMPONENTS))
        if unknown:
            raise ValueError(f"NPM {self.name} has unknown components {unknown}: the known components are {COMPONENTS}")
        for needed in ("protein", self.protein_cap_component):
            if needed not in names:
                raise ValueError(f"NPM {self.name} has no {needed} component, which the protein cap requires")

    def component(self, name: str) -> Component:
        """Return the component called `name`.

        Args:
            name (str): The component, e.g. "energy" or "free_sugars".

        Returns:
            Component: The component.

        Raises:
            KeyError: If the scheme has no such component.
        """
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(f"NPM {self.name} has no {name!r} component: it has {[c.name for c in self.components]}")

    @property
    def score_range(self) -> tuple[int, int]:
        """The lowest and highest possible scores, i.e. all C points and no A points, and vice versa."""
        a = sum(c.max_points for c in self.components if c.kind == "A")
        c = sum(c.max_points for c in self.components if c.kind == "C")
        return -c, a


def get_scheme(scheme: str | int | Scheme) -> Scheme:
    """Return a scheme by its name, or the scheme itself if given one.

    Args:
        scheme (str | int | Scheme): 2004 or 2018 (as an int or str), or a `Scheme`, e.g. one from `load_scheme`.

    Returns:
        Scheme: The scheme.

    Raises:
        ValueError: If there is no scheme by that name.
    """
    if isinstance(scheme, Scheme):
        return scheme
    scheme = str(scheme)
    if scheme not in SCHEMES:
        raise ValueError(f"No NPM scheme {scheme!r}: the schemes are {SCHEMES}")
    return _builtin_scheme(scheme)


def load_scheme(path: str | Path) -> Scheme:
    """Load a scheme from a YAML file with the same layout as `ahl_npm/schemes/npm_2018.yaml`, e.g. to vary thresholds.

    Args:
        path (str | Path): The YAML file.

    Returns:
        Scheme: The scheme, which can be passed anywhere a scheme's name can.
    """
    return _parse_scheme(yaml.safe_load(Path(path).read_text()))


@cache
def _builtin_scheme(name: str) -> Scheme:
    return _parse_scheme(yaml.safe_load((resources.files("ahl_npm") / "schemes" / f"npm_{name}.yaml").read_text()))


def _parse_scheme(config: dict) -> Scheme:
    """Build a scheme from a parsed YAML, converting thresholds to the units inputs are given in."""
    components = tuple(
        Component(
            name=component,
            kind=spec["kind"],
            # Divide rather than multiply by the reciprocal, so that 90 mg is exactly the float 0.09 g.
            thresholds=tuple(t / _UNIT_DIVISORS[spec["unit"]] for t in spec["thresholds"]),
            points=tuple(spec.get("points", range(1, len(spec["thresholds"]) + 1))),
        )
        for component, spec in config["components"].items()
    )
    return Scheme(
        name=config["name"],
        source=config["source"],
        components=components,
        protein_cap_a_points=config["protein_cap"]["a_points"],
        protein_cap_component=config["protein_cap"]["component"],
        protein_cap_points=config["protein_cap"]["points"],
        less_healthy_food=config["less_healthy"]["food"],
        less_healthy_drink=config["less_healthy"]["drink"],
    )
