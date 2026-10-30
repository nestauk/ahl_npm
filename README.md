# AHL NPM Scoring

Two definitions of the UK's Nutrient Profiling Model, [NPM 2004/05](https://www.gov.uk/government/publications/the-nutrient-profiling-model) and [NPM 2018](https://www.gov.uk/government/publications/nutrient-profiling-model-2018/nutrient-profiling-model-2018-technical-guidance), scored on `numpy` arrays or `polars` frames.

## Installing

The package is installable from [PyPI](https://pypi.org/project/ahl-npm/) on Python 3.10+ with `pip`, `uv`, etc.:

```bash
uv add ahl-npm
```

## Usage

```python
import ahl_npm as npm

# numpy: numbers or arrays per 100g, which broadcast together (e.g. one panel against multiple possible free sugars values)
scored = npm.score(
    scheme=2018, energy_kj=1500, satfat=4.5, free_sugars=[2.0, 6.5], salt=1.25, protein=10, fibre=3, fvns_pct=0
)
scored["npm"]  # array([11., 15.]), with each component's points, the A and C points and whether protein counts

# polars: `columns` maps each input to a column or an expression; without it, the frame's columns named as inputs
scored = npm.score_frame(
    frame=df,
    scheme=2004,
    columns={
        "energy_kcal": "kcal_per_100g",
        "satfat": "satfat_per_100g",
        "sugars": "sugar_per_100g",
        "sodium": "sodium_per_100g",
        "protein": "protein_per_100g",
        "fibre": "fibre_per_100g",
        "fvn_pct": pl.col("fvn_share") * 100,
    },
)  # adds energy_points_2004, ..., a_points_2004, c_points_2004, protein_counted_2004, npm_2004, cnpm_2004
```

`outputs` picks which columns to add and, when passed as a mapping, renames them (between `prefix` and `suffix`), e.g. `outputs={"npm": "npm_score", "cnpm": "cnpm_score"}` only adds `npm_score_2004` and `cnpm_score_2004`. Inputs a scheme doesn't use are ignored, so the same mapping can be used for both schemes.

`score_expr` returns the same scoring as a struct expression, and `band` and `band_expr` give the points for a single component, e.g. `npm.band(2018, "free_sugars", values)`.

### Inputs

All inputs are densities (g/100g) as sold. Where a component can be scored from more than one input, each row uses the first one available: the quantity the scheme scores, then a converted stand-in, then the component's points if given.

| Input                      | Unit | Scored by                                                                             |
| -------------------------- | ---- | ------------------------------------------------------------------------------------- |
| `energy_kj`, `energy_kcal` | kJ   | both, using kJ if given, else kcal x 4.184                                            |
| `satfat`                   | g    | both                                                                                  |
| `sugars`                   | g    | NPM 2004/05 (total sugars)                                                            |
| `free_sugars`              | g    | NPM 2018                                                                              |
| `sodium`, `salt`           | g    | NPM 2004/05 scores sodium (else salt / 2.5), NPM 2018 scores salt (else sodium x 2.5) |
| `protein`                  | g    | both                                                                                  |
| `fibre`                    | g    | both (AOAC method)                                                                    |
| `fvn_pct`                  | %    | NPM 2004/05 (fruit, vegetables and nuts)                                              |
| `fvns_pct`                 | %    | NPM 2018 (fruit, vegetables, nuts and seeds)                                          |
| `<component>_points`       |      | both, for points already scored, as an alternative to the densities above             |

Points inputs are named the same as the outputs, so a scoring's points can be passed straight back in, and are checked against the points the component can score under the scheme. NPM 2018 includes seeds, so it only uses `fvns_pct` or `fvns_points`: if you only have the NPM 2004/05 share, pass it as `fvns_pct` too. NPM 2018 also needs free sugars. Scoring total sugars as free sugars isn't supported, as it overestimates NPM 2018 scores, but can be done by passing total sugars as `free_sugars`.

Salt is sodium x 2.5, as UK labelling law defines it ([Regulation (EU) No 1169/2011, Annex I](https://www.legislation.gov.uk/eur/2011/1169/annex/I), as retained in UK law).

### Outputs

Each component's points (`<component>_points`), `a_points`, `c_points`, `protein_counted`, `npm` and `cnpm` (70 - 2 x `npm`). The numpy functions return floats with NaN where unknown (or integers with `strict=True`, which raises on missing inputs), and the polars functions return integers (a boolean for `protein_counted`) with nulls where unknown. Protein stops counting from 11 A points, unless fruit, vegetables and nuts score 5 points.

### Checks and warnings

The most likely mistake is an input in the wrong units, so each frame scoring logs (at INFO, to the `ahl_npm` logger) which column or expression each component reads, in what units, and with any conversion. `describe_inputs` gives the same description for any inputs:

```text
NPM 2004 reads, per 100 g:
  energy: energy_kcal <- "kcal_per_100g", kcal x 4.184 to kJ
  sodium: sodium <- "sodium_per_100g", g; else salt <- "salt_per_100g", g / 2.5 to sodium
  fvn: fvn_pct <- [(col("fvn_share")) * (dyn int: 100)], %
  ...
```

- `ImpossibleInputWarning` warns of implausible density values, i.e. below 0, or above 100g, 100% or the energy of pure fat (3,700 kJ or 900 kcal), which usually means an input is in the wrong units (e.g. sodium in mg read as g).
- `MissingInputWarning` summarises missing inputs (NaN, or null in polars). A missing input scores NaN points rather than the maximum, so any score it feeds into is missing too.
- `warn=False` turns both warnings off. A frame only warns once, for the first batch of rows with problems.

### Other functions

- `to_cnpm` and `from_cnpm` convert between NPM and cNPM (70 - 2 x the score, s.t. higher is healthier) for numbers, arrays, and polars expressions or series. `clip=True` clips cNPM to between 0 and 100, and `change=True` converts changes in scores (without the offset).
- `less_healthy(scheme, npm, drink)` gives whether each product is less healthy, i.e. a food scoring 4 or more, or a drink scoring 1 or more. `drink` must be booleans with none missing (in polars, a null gives a null). Whether a product is in scope of HFSS legislation is a separate question.
- `score_range(scheme, low, high, **fixed)` gives the lowest and highest scores for inputs anywhere between their `low` and `high` values. The score never decreases as an A input increases or a C input decreases, so only two scorings are needed, however many inputs are uncertain.

## Development Setup

This project uses [`uv`](https://docs.astral.sh/uv/) and `direnv`. To set up a development environment, run:

```bash
direnv allow
uv sync
uv run pre-commit install --install-hooks
uv run pytest
```

`uv run tox` runs the tests under each Python version from 3.10 to 3.15, and (`lowest`) with the oldest numpy, polars and PyYAML versions allowed. CI runs these on every pull request and push to `dev` or `main`, and weekly.
