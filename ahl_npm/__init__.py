"""ahl_npm: the UK's Nutrient Profiling Model (NPM 2004/05 and NPM 2018), scored on numpy arrays or polars frames."""

__version__ = "1.0.0"

from ahl_npm.constants import KJ_PER_KCAL, SALT_PER_SODIUM
from ahl_npm.frames import band_expr, score_expr, score_frame
from ahl_npm.schemes import COMPONENTS, SCHEMES, Component, Scheme, get_scheme, load_scheme
from ahl_npm.scoring import (
    INPUTS,
    ImpossibleInputWarning,
    MissingInputWarning,
    band,
    describe_inputs,
    output_names,
    score,
    score_range,
)
from ahl_npm.transforms import from_cnpm, less_healthy, to_cnpm

__all__ = [
    "COMPONENTS",
    "INPUTS",
    "ImpossibleInputWarning",
    "KJ_PER_KCAL",
    "SALT_PER_SODIUM",
    "SCHEMES",
    "Component",
    "MissingInputWarning",
    "Scheme",
    "band",
    "band_expr",
    "describe_inputs",
    "from_cnpm",
    "get_scheme",
    "less_healthy",
    "load_scheme",
    "output_names",
    "score",
    "score_expr",
    "score_frame",
    "score_range",
    "to_cnpm",
]
