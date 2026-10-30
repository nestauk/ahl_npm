# AHL NPM Scoring

The UK's Nutrient Profiling Model, [NPM 2004/05](https://www.gov.uk/government/publications/the-nutrient-profiling-model) and [NPM 2018](https://www.gov.uk/government/publications/nutrient-profiling-model-2018/nutrient-profiling-model-2018-technical-guidance), scored on numpy arrays or polars frames, for every project in Nesta's A Healthy Life (AHL) mission to share. Each scheme is a YAML file of the technical guidance's tables (`ahl_npm/schemes`), and the scoring is written once, in numpy (`ahl_npm/scoring.py`): the polars functions run it on each batch of rows, so a frame scores exactly as arrays do, eager, lazy or streaming. It scores exactly as `ahl_core_data`'s `calculate_scores` did, threshold for threshold.

## Installing

`ahl-npm` is on [PyPI](https://pypi.org/project/ahl-npm/):

```bash
uv add ahl-npm
```

## Using it

```python
import ahl_npm as npm

# numpy: numbers or arrays per 100 g, which broadcast together (one panel against many draws of free sugars, say)
scored = npm.score(2018, energy_kj=1500, satfat=4.5, free_sugars=[2.0, 6.5], salt=1.25, protein=10, fibre=3, fvns_pct=0)
scored["npm"]  # array([11., 15.]), with each component's points, the A and C points and whether protein counts

# polars: `columns` maps each input to a column or an expression; without it, the frame's columns named as inputs
scored = npm.score_frame(
    frame,
    2004,
    {
        "energy_kcal": "kcal_per_100g",
        "satfat": "satfat_per_100g",
        "sugars": "sugar_per_100g",
        "sodium": "sodium_per_100g",
        "protein": "protein_per_100g",
        "fibre": "fibre_per_100g",
        "fvn_pct": pl.col("fvn_share") * 100,
    },
    cnpm=True,
)  # adds energy_points_2004, ..., a_points_2004, c_points_2004, protein_counted_2004, npm_2004, cnpm_2004
```

`outputs` picks which columns to add and, as a mapping, what to call them (between `prefix` and `suffix`): `outputs={"npm": "npm_score", "cnpm": "cnpm_score"}` adds `npm_score_2004` and `cnpm_score_2004` alone. A scheme ignores the inputs it doesn't read, so one mapping serves both schemes, a column only the other reads needn't exist.

`score_expr` gives the same scoring as a struct expression, and `band` and `band_expr` the points one component gives (`npm.band("2018", "free_sugars", values)`). Points already scored go in as `<component>_points`, alone or alongside values:

```python
npm.score_frame(retailer, "2004")  # a frame of energy_points, satfat_points, ..., fvn_points
```

### Inputs

All per 100 g as sold. A component can score from more than one input, and each row takes the first it has: the quantity the scheme scores, then a stand-in converted to it, then the component's points as given.

| Input                      | Unit | Scored by                                                                      |
| -------------------------- | ---- | ------------------------------------------------------------------------------ |
| `energy_kj`, `energy_kcal` | kJ   | both, kJ first, else kcal x 4.184                                              |
| `satfat`                   | g    | both                                                                           |
| `sugars`                   | g    | NPM 2004/05 (total sugars)                                                     |
| `free_sugars`              | g    | NPM 2018                                                                       |
| `sodium`, `salt`           | g    | NPM 2004/05 scores sodium (else salt / 2.5), NPM 2018 salt (else sodium x 2.5) |
| `protein`                  | g    | both                                                                           |
| `fibre`                    | g    | both, by the AOAC method                                                       |
| `fvn_pct`                  | %    | NPM 2004/05: the fruit, vegetable and nut share                                |
| `fvns_pct`                 | %    | NPM 2018: the fruit, vegetable, nut and seed share                             |
| `<component>_points`       |      | both: a component's points already scored, such as a retailer's own            |

The points inputs are named as the outputs are (`energy_points`, ..., `fvn_points`), so a scoring's points can go straight back in, and are checked to be ones the component can score under the scheme (NPM 2018's fibre points up to 10 are an error under NPM 2004/05). Kantar's fruit, vegetable and nut category lookup is `fvn_points`. NPM 2018 counts seeds as well, so it takes `fvns_pct` or `fvns_points` and never NPM 2004/05's share in their place: a project with only that share passes it as both, knowingly.

Salt is sodium x 2.5, as UK labelling law defines it and so as the salt on a pack is: the 2.54 of sodium chloride's mass would hold only if all of a product's sodium came from table salt.

NPM 2018 needs free sugars, which Kantar doesn't give us yet. Scoring total sugars as free sugars isn't supported, as it overstates NPM 2018 scores badly, but nothing stops a project doing so knowingly: `columns={"free_sugars": "sugar_per_100g"}`.

### Outputs

Each component's points (`<component>_points`), `a_points`, `c_points` (as counted), `protein_counted` and `npm`: floats from numpy, NaN where unknown (or, with `strict=True`, integers, and an error where an input is missing), and integers (a boolean for `protein_counted`) from polars, null where unknown. Protein stops counting from 11 A points unless fruit, vegetables and nuts score 5, so a row whose protein can't count scores without it.

### Checks on the inputs

The likeliest mistake is an input in the wrong units, so each frame's scoring logs, at INFO to the `ahl_npm` logger, what every component reads, from which column or expression, in what units and with what conversion, for you to check (`describe_inputs` gives the same for any inputs):

```text
NPM 2004 reads, per 100 g:
  energy: energy_kcal <- "kcal_per_100g", kcal x 4.184 to kJ
  sodium: sodium <- "sodium_per_100g", g; else salt <- "salt_per_100g", g / 2.5 to sodium
  fvn: fvn_pct <- [(col("fvn_share")) * (dyn int: 100)], %
  ...
```

Beyond that, it guesses at nothing, but warns (`ImpossibleInputWarning`) of values no 100 g of anything could hold, below 0 or above 100 g, 100% or pure fat's 3,700 kJ (900 kcal), as only the wrong units (sodium in mg read as g) or wrong data give them. A missing input (NaN, or null in polars) scores NaN points, never the most points, and the scores it feeds are missing too, which a `MissingInputWarning` sums up: how many scores are missing and how many rows each component was missing from. Points given are checked to be ones their component can score under the scheme. `warn=False` turns the warnings off. A frame's scoring warns once, of the first batch of rows it finds wanting: the whole frame, unless a streaming collect scores it in batches.

### What follows from a score

- `to_cnpm(npm, clip=False, change=False)`: cNPM, 70 - 2 x the score, from -10 to 100 under NPM 2004/05 and to 110 under NPM 2018; `clip=True` keeps it from 0 to 100, and `change=True` converts a change in score (-2 x it, no offset). `from_cnpm` goes back. Both take numbers, arrays and polars expressions or series.
- `less_healthy(scheme, npm, drink)`: a food scores less healthy from 4, a drink from 1. Whether each product is a drink must be known, as booleans (in polars, a null gives a null), and a missing score gives a missing answer. A product is HFSS when it is also in scope of the legislation, which is for each project to decide.
- `score_range(scheme, low, high, **fixed)`: the lowest and highest scores for inputs anywhere between their lows and highs. A score never falls as an A input rises or a C input falls, so these are two scorings, at opposite corners, however many inputs are uncertain.

### Schemes of your own

`load_scheme(path)` reads a YAML laid out as `ahl_npm/schemes/npm_2018.yaml` is, such as NPM 2018 with other thresholds for a sensitivity analysis, and the `Scheme` it gives goes anywhere a scheme's name does. Its components must be among the ten the built-in schemes have.

### Scoring at thresholds

A value scores a threshold's point only above it, as the guidance has it, and by more than 1e-4 + 1e-5 x the threshold, so float error (100 x a total / a mass, kcal x 4.184) never crosses one. That is `np.isclose`'s tolerance, which `ahl_core_data` has always snapped values to thresholds with.

### Speed

On 5 million rows, both schemes, on an 8-core laptop: `score` takes 0.5 s, `score_frame` 0.8 s eager and 0.3 s streaming (which runs batches in parallel), against 3.3 s for `ahl_core_data` v1.5.0's `calculate_scores`. A single row takes about 40 µs and 4,000 draws about 0.3 ms, two to three times faster than the free sugars project's numpy port.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) and `direnv`. Its packages all come from PyPI, so `.envrc` unsets any `UV_INDEX` you have set. Then run:

```bash
direnv allow
uv sync
uv run pre-commit install --install-hooks
uv run pytest
```

`uv run tox` runs the tests under each Python from 3.10 to 3.15 and, as `lowest`, under Python 3.10 with the oldest numpy, polars and PyYAML `ahl-npm` allows, each resolving its dependencies afresh, as a project installing `ahl-npm` would; `uv run tox -e py313` runs one. CI runs them all on every pull request and push to `dev` or `main`, and weekly, to catch a new numpy or polars release breaking `ahl-npm` before a project upgrading to it does.

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

### Versioning

We use [Semantic Versioning](https://semver.org/). The version is `__version__` in `ahl_npm/__init__.py`, which `hatchling` reads at build. To release:

1. Bump `__version__` in a release PR into `dev` (`release/vX.Y.Z`), and merge it once approved and green.
2. [Publish a GitHub release](https://github.com/nestauk/ahl_npm/releases/new) from `dev`, tagged `v` and the version, such as `v1.0.0`.
3. Publishing it runs [`publish-pypi.yml`](https://github.com/nestauk/ahl_npm/blob/dev/.github/workflows/publish-pypi.yml), which builds the tag and publishes it to PyPI, refusing a tag that isn't the version. PyPI trusts the workflow as the project's publisher (trusted publishing: owner `nestauk`, repository `ahl_npm`, workflow `publish-pypi.yml`, environment `pypi`), so there is no token to keep. PyPI never takes a version twice, so a published version is final.
