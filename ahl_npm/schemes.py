"""NPM schemes as data: each component's thresholds and points, the protein cap and the less healthy cut-offs.

Each scheme is a YAML file in `ahl_npm/schemes`, read once into a `Scheme`; `load_scheme` reads one of your own.
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

# The components a scheme can have, which `ahl_npm.scoring` knows the inputs of: `fvn` is NPM 2004/05's fruit,
# vegetables and nuts, `fvns` NPM 2018's fruit, vegetables, nuts and seeds.
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

# What each unit a scheme's thresholds come in divides by to give the units inputs come in: kJ, g or % per 100 g.
_UNIT_DIVISORS: dict[str, int] = {"kJ": 1, "g": 1, "mg": 1000, "%": 1}

# Below this many values, one binary search bands them faster than a comparison per threshold does.
_SEARCH_BELOW: int = 512


@dataclass(frozen=True)
class Component:
    """One part of a scheme's score, and the points it gives a value per 100 g.

    Attributes:
        name (str): The component, such as "energy" or "free_sugars".
        kind (str): "A" for a component counting towards the score, "C" for one counting against it.
        thresholds (tuple[float, ...]): The values to be above for each point, in kJ, g or % per 100 g.
        points (tuple[int, ...]): The points for being above each threshold.
    """

    name: str
    kind: Literal["A", "C"]
    thresholds: tuple[float, ...]
    points: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check the thresholds and points make a step function."""
        if self.kind not in ("A", "C"):
            raise ValueError(f"{self.name}: kind must be 'A' or 'C', not {self.kind!r}")
        if len(self.points) != len(self.thresholds):
            raise ValueError(f"{self.name}: {len(self.points)} points for {len(self.thresholds)} thresholds")
        if any(b <= a for a, b in pairwise(self.thresholds)):
            raise ValueError(f"{self.name}: thresholds must rise")
        if any(b <= a for a, b in pairwise((0, *self.points))):
            raise ValueError(f"{self.name}: points must rise from 0")

    @cached_property
    def edges(self) -> np.ndarray:
        """The thresholds as values are compared with them, each a hair higher so float error never crosses one."""
        thresholds = np.asarray(self.thresholds, dtype=float)
        return thresholds + BAND_ATOL + BAND_RTOL * np.abs(thresholds)

    @cached_property
    def point_values(self) -> np.ndarray:
        """The points for being above none, one, two, ... of the thresholds: every value the component can score."""
        return np.asarray((0, *self.points), dtype=float)

    @cached_property
    def _band_edges(self) -> np.ndarray:
        # The edges then infinity, which every number is at or below and NaN sorts past.
        return np.append(self.edges, np.inf)

    @cached_property
    def _band_points(self) -> np.ndarray:
        # The points for being above none, one, two, ... of the edges, then NaN for a missing value.
        return np.append(self.point_values, np.nan)

    def band(self, values: np.ndarray) -> np.ndarray:
        """Return the points float values per 100 g score: those for the highest threshold each is above.

        Args:
            values (np.ndarray): Float values per 100 g, NaN where missing.

        Returns:
            np.ndarray: The points, as floats, NaN where a value is.
        """
        if values.size < _SEARCH_BELOW:
            return self._band_points[np.searchsorted(self._band_edges, values, side="left")]
        # On many values, counting the edges each is above, a comparison per edge, beats a binary search over so few.
        above = np.zeros(values.shape, dtype=np.uint8)
        for edge in self.edges:
            np.add(above, values > edge, out=above, casting="unsafe")
        above[np.isnan(values)] = len(self.edges) + 1
        return self._band_points[above]

    @property
    def max_points(self) -> int:
        """The most points the component can give."""
        return self.points[-1]


@dataclass(frozen=True)
class Scheme:
    """An NPM scheme: its components, when protein counts, and the scores from which food and drink are less healthy.

    Attributes:
        name (str): The scheme, "2004" or "2018".
        source (str): Where its technical guidance is published.
        components (tuple[Component, ...]): Its components, A then C.
        protein_cap_a_points (int): The A points from which protein stops counting...
        protein_cap_component (str): ... unless this component, `fvn` or `fvns`, scores...
        protein_cap_points (int): ... at least these points.
        less_healthy_food (int): The score from which a food is less healthy.
        less_healthy_drink (int): The score from which a drink is less healthy.
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
        """Check the scheme's components are ones inputs are known for, with the two its protein cap names."""
        names = [c.name for c in self.components]
        unknown = sorted(set(names) - set(COMPONENTS))
        if unknown:
            raise ValueError(f"NPM {self.name} has components {unknown}, beyond the ones scored: {COMPONENTS}")
        for needed in ("protein", self.protein_cap_component):
            if needed not in names:
                raise ValueError(f"NPM {self.name} has no {needed} component, which the protein cap needs")

    def component(self, name: str) -> Component:
        """Return the component called `name`.

        Args:
            name (str): The component, such as "energy" or "free_sugars".

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
        """The lowest and highest scores the scheme can give: all C points and no A points, then the reverse."""
        a = sum(c.max_points for c in self.components if c.kind == "A")
        c = sum(c.max_points for c in self.components if c.kind == "C")
        return -c, a


def get_scheme(scheme: str | int | Scheme) -> Scheme:
    """Return a scheme, by name or as given.

    Args:
        scheme (str | int | Scheme): 2004 or 2018 (or "2004" or "2018"), or a `Scheme`, such as one from `load_scheme`.

    Returns:
        Scheme: The scheme.

    Raises:
        ValueError: If there is no scheme by that name.
    """
    if isinstance(scheme, Scheme):
        return scheme
    scheme = str(scheme)
    if scheme not in SCHEMES:
        raise ValueError(f"No NPM scheme {scheme!r}: the schemes are {SCHEMES}, or load your own with `load_scheme`")
    return _builtin_scheme(scheme)


def load_scheme(path: str | Path) -> Scheme:
    """Read a scheme from a YAML file laid out as `ahl_npm/schemes/npm_2018.yaml` is, such as one with other thresholds.

    Args:
        path (str | Path): The YAML file.

    Returns:
        Scheme: The scheme, to pass anywhere a scheme's name goes.
    """
    return _parse_scheme(yaml.safe_load(Path(path).read_text()))


@cache
def _builtin_scheme(name: str) -> Scheme:
    return _parse_scheme(yaml.safe_load((resources.files("ahl_npm") / "schemes" / f"npm_{name}.yaml").read_text()))


def _parse_scheme(config: dict) -> Scheme:
    """Build a scheme from its YAML's contents, with thresholds in the units inputs come in."""
    components = tuple(
        Component(
            name=component,
            kind=spec["kind"],
            # Divided, not multiplied by a reciprocal, so 90 mg is exactly the float 0.09 g.
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
