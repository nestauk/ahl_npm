import inspect
import itertools
import warnings

import numpy as np
import pytest

from ahl_npm import (
    INPUTS,
    ImpossibleInputWarning,
    MissingInputWarning,
    band,
    describe_inputs,
    get_scheme,
    score,
    score_range,
)

# A product worked by hand. NPM 2004/05: energy 1500 kJ 4 points, saturated fat 4.5 g 4, sugars 20 g 4, sodium 0.5 g 5,
# so 17 A points, and protein can't count; fibre 3 g 3, so 17 - 3 = 14. NPM 2018: 4, 4, free sugars 10, salt 1.25 g 5,
# so 23; fibre 3.0 g is not above 3.0, so 4 points, and 23 - 4 = 19.
PANEL = {"energy_kj": 1500, "satfat": 4.5, "sodium": 0.5, "protein": 10, "fibre": 3}


def test_worked_example_2004() -> None:
    """Every part of the 2004/05 scoring matches the hand-worked product."""
    scored = score("2004", **PANEL, sugars=20, fvn_points=0)
    assert {k: v.item() for k, v in scored.items()} == {
        "energy_points": 4,
        "satfat_points": 4,
        "sugars_points": 4,
        "sodium_points": 5,
        "protein_points": 5,
        "fibre_points": 3,
        "fvn_points": 0,
        "a_points": 17,
        "c_points": 3,
        "protein_counted": 0,
        "npm": 14,
    }


def test_worked_example_2018() -> None:
    """Every part of the 2018 scoring matches the hand-worked product, salt from sodium at 2.5."""
    scored = score("2018", **PANEL, free_sugars=20, fvns_pct=0)
    assert {k: v.item() for k, v in scored.items()} == {
        "energy_points": 4,
        "satfat_points": 4,
        "free_sugars_points": 10,
        "salt_points": 5,
        "protein_points": 5,
        "fibre_points": 4,
        "fvns_points": 0,
        "a_points": 23,
        "c_points": 4,
        "protein_counted": 0,
        "npm": 19,
    }


def test_points_need_a_value_above_each_threshold() -> None:
    """A value at a threshold, or above it by float error, scores the band below."""
    values = [0, 1.0, 1.00005, 1.0002, 10, 10.5]
    assert band("2004", "satfat", values).tolist() == [0, 0, 0, 1, 9, 10]


def test_tolerance_grows_with_the_threshold() -> None:
    """As in ahl_core_data, the tolerance is 1e-4 plus 1e-5 x the threshold: about 0.03 kJ at 3150 kJ."""
    assert band("2018", "energy", [3150.0, 3150.02, 3150.04]).tolist() == [9, 9, 10]


@pytest.mark.parametrize("component", ["energy", "salt", "fvns"])
def test_few_and_many_values_band_alike(component: str) -> None:
    """Banding takes a binary search for a few values and a comparison per threshold for many: they agree."""
    edges = np.asarray(get_scheme("2018").component(component).thresholds)
    values = np.concatenate([edges, edges + 1e-5, edges + 1e-3, edges - 1e-3, [np.nan, np.inf, -np.inf, -1.0]])
    values = np.random.default_rng(3).permutation(np.tile(values, 30))  # well past the 512 the search stops at
    many = band("2018", component, values, warn=False)  # -1 and infinity, which no 100 g holds, on purpose
    few = np.concatenate([band("2018", component, chunk, warn=False) for chunk in np.array_split(values, 50)])
    np.testing.assert_array_equal(many, few)


def test_missing_values_score_nan() -> None:
    """A missing value scores NaN points, not the most points."""
    assert np.isnan(band("2004", "fibre", [np.nan])).all()


def test_protein_cap() -> None:
    """From 11 A points protein stops counting, unless fruit, vegetables and nuts score 5."""
    capped = {"energy_kj": 3400, "satfat": 1.5, "sugars": 0, "sodium": 0, "protein": 10, "fibre": 0}  # 10 + 1 A points
    assert score("2004", **capped, fvn_points=[0, 2, 5])["npm"].tolist() == [11, 9, 1]
    assert score("2004", **{**capped, "satfat": 0.5}, fvn_points=0)["npm"].tolist() == [5]


def test_protein_is_only_needed_where_it_counts() -> None:
    """A row whose protein can't count scores without it, while one whose protein can is NaN."""
    capped = {"energy_kj": 3400, "satfat": [1.5, 0.5], "sugars": 0, "sodium": 0, "fibre": 0, "fvn_points": 0}
    with pytest.warns(MissingInputWarning, match=r"1 of 2 scores \(50.0%\).*protein 2\); protein is only needed"):
        scored = score("2004", **capped, protein=np.nan)
    assert scored["npm"][0] == 11
    assert np.isnan(scored["npm"][1])
    assert scored["protein_counted"].tolist() == [0, 1]


def test_missing_inputs_warn_with_a_summary() -> None:
    """The warning counts the missing scores and the rows each component is missing from."""
    with pytest.warns(MissingInputWarning, match=r"NPM 2018: 2 of 3 scores \(66.7%\).*free_sugars 1, fibre 1\)"):
        score("2018", **{**PANEL, "fibre": [3, np.nan, 3]}, free_sugars=[1, 1, np.nan], fvns_points=0)


def test_warnings_can_be_turned_off() -> None:
    """`warn=False` scores missing inputs in silence."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score("2018", **PANEL, free_sugars=np.nan, fvns_points=0, warn=False)


def test_stand_ins_fill_rows_missing_the_scored_input() -> None:
    """Each row takes the input the scheme scores where it has it: kJ over kcal, sodium over salt, salt over sodium."""
    base = {"satfat": 0, "sugars": 0, "protein": 0, "fibre": 0, "fvn_points": 0}
    energy = score("2004", **base, sodium=0, energy_kj=[300, np.nan], energy_kcal=[500, 100])["energy_points"]
    assert energy.tolist() == [0, 1]  # 300 kJ; 100 kcal is 418.4 kJ
    sodium = score("2004", **base, energy_kj=0, sodium=[0.1, np.nan], salt=[0.1, 0.5])["sodium_points"]
    assert sodium.tolist() == [1, 2]  # 0.1 g sodium; 0.5 g salt is 0.2 g sodium
    salt = score("2018", **{**base, "free_sugars": 0, "fvns_points": 0}, energy_kj=0, sodium=0.2)["salt_points"]
    assert salt.tolist() == [1]  # 0.5 g salt, not above 0.5: salt is sodium x 2.5, not 2.54


def test_npm_2018_needs_free_sugars() -> None:
    """NPM 2018 won't score total sugars as free sugars: a caller who wants that must pass them as free sugars."""
    with pytest.raises(ValueError, match="NPM 2018 needs free_sugars"):
        score("2018", **PANEL, sugars=20, fvns_points=0)


def test_inputs_broadcast() -> None:
    """One panel scores against many draws of free sugars, and products against draws."""
    draws = np.linspace(0, 12, 1000)
    assert score("2018", **PANEL, free_sugars=draws, fvns_points=0)["npm"].shape == (1000,)
    by_product = score("2018", **PANEL, free_sugars=draws[None, :], fvns_points=np.array([[0], [5]]))
    assert by_product["npm"].shape == (2, 1000)


@pytest.mark.parametrize("scheme", ["2004", "2018"])
def test_points_given_back_score_the_same(scheme: str) -> None:
    """A scoring's points, given back as inputs, rebuild it exactly."""
    rng = np.random.default_rng(2)
    values = {
        "energy_kj": rng.uniform(0, 3600, 5000),
        "satfat": rng.uniform(0, 11, 5000),
        "sugars": rng.uniform(0, 50, 5000),
        "free_sugars": rng.uniform(0, 11, 5000),
        "salt": rng.uniform(0, 2.6, 5000),
        "protein": rng.uniform(0, 10, 5000),
        "fibre": rng.uniform(0, 7, 5000),
        "fvn_pct": rng.uniform(0, 100, 5000),
        "fvns_pct": rng.uniform(0, 100, 5000),
    }
    scored = score(scheme, **values)
    rebuilt = score(scheme, **{name: p for name, p in scored.items() if name in INPUTS})
    for name, expected in scored.items():
        np.testing.assert_array_equal(rebuilt[name], expected)


def test_points_stand_in_where_values_are_missing() -> None:
    """Points fill the rows a component's values are missing from, and values and points mix across components."""
    scored = score("2004", **{**PANEL, "fibre": [3, np.nan]}, fibre_points=[0, 3], sugars_points=4, fvn_points=0)
    assert scored["fibre_points"].tolist() == [3, 3]  # the value where given, the points where not
    assert scored["npm"].tolist() == [14, 14]


def test_points_must_be_the_schemes() -> None:
    """Points a component can't score, such as NPM 2018 fibre points under NPM 2004/05, are an error."""
    with pytest.raises(ValueError, match=r"fibre_points must be NPM 2004 fibre points, one of \[0 1 2 3 4 5\]"):
        score("2004", **PANEL, sugars=20, fvn_points=0, fibre_points=[2, 7])
    with pytest.raises(ValueError, match="fvn_points"):
        score("2004", **PANEL, sugars=20, fvn_points=3)


def test_score_range_brackets_every_input_in_the_box() -> None:
    """The two scorings bound the score anywhere between the inputs' lows and highs, and are what the corners give."""
    rng = np.random.default_rng(0)
    names = ("energy_kj", "satfat", "free_sugars", "salt", "protein", "fibre", "fvns_pct")
    scale = np.array([3500, 10, 12, 2.5, 10, 7, 100])
    for _ in range(50):
        ends = np.sort(rng.uniform(0, 1, (2, len(names))) * scale, axis=0)
        low, high = dict(zip(names, ends[0], strict=True)), dict(zip(names, ends[1], strict=True))
        lowest, highest = score_range("2018", low, high)
        inside = score(
            "2018", **{n: rng.uniform(lo, hi, 500) for n, lo, hi in zip(names, ends[0], ends[1], strict=True)}
        )
        assert (lowest["npm"] <= inside["npm"]).all() and (inside["npm"] <= highest["npm"]).all()
        corners = [score("2018", **dict(zip(names, c, strict=True)))["npm"].item() for c in itertools.product(*ends.T)]
        assert (lowest["npm"].item(), highest["npm"].item()) == (min(corners), max(corners))


def test_npm_2018_scores_seeds_too_so_takes_no_fvn_in_their_place() -> None:
    """NPM 2018 scores seeds with fruit, vegetables and nuts: a share without them stands in only if passed as one."""
    with pytest.raises(ValueError, match="fvns_pct or fvns_points"):
        score("2018", **PANEL, free_sugars=0, fvn_pct=50)
    assert score("2018", **PANEL, free_sugars=0, fvns_pct=50)["fvns_points"].item() == 1


def test_signature_takes_every_input() -> None:
    """`score`'s keyword arguments are `INPUTS`, in order."""
    params = [p for p in inspect.signature(score).parameters if p not in ("scheme", "strict", "warn")]
    assert tuple(params) == INPUTS


def test_strict_scores_are_integers_and_refuse_missing_inputs() -> None:
    """`strict` returns integers where every input is known, and refuses a missing one instead of leaving NaN."""
    scored = score(2004, **PANEL, sugars=20, fvn_points=0, strict=True)
    assert scored["npm"].dtype.kind == "i" and scored["npm"].tolist() == [14]
    with pytest.raises(ValueError, match="fibre 1.*strict refuses"):
        score(2004, **{**PANEL, "fibre": np.nan}, sugars=20, fvn_points=0, strict=True)
    assert band(2018, "free_sugars", [0.5, 9.4], strict=True).tolist() == [0, 10]
    with pytest.raises(ValueError, match="1 free_sugars values are missing"):
        band(2018, "free_sugars", [np.nan], strict=True)


def test_values_no_100g_could_hold_warn_in_numpy_too() -> None:
    """Below 0 or beyond what 100 g holds warns, the scores still made; inputs a scheme doesn't read go unchecked."""
    with pytest.warns(ImpossibleInputWarning, match="salt 1 above 100 g"):
        score(2018, **PANEL, salt=250.0, free_sugars=0, fvns_points=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        score(2004, **PANEL, sugars=20, fvn_points=0, free_sugars=-5.0)  # 2004 reads no free sugars


def test_a_value_given_once_counts_once_however_it_broadcasts() -> None:
    """One impossible salt scored against 1000 draws of free sugars is one impossible value, not 1000."""
    with pytest.warns(ImpossibleInputWarning, match="salt 1 above 100 g$"):
        score(2018, **PANEL, salt=250.0, free_sugars=np.linspace(0, 12, 1000), fvns_points=0)


def test_describe_inputs_names_each_components_reads() -> None:
    """The description lists, per component, every input it would read in the order a row takes them."""
    text = describe_inputs(2018, {"energy_kj": "kj", "energy_kcal": "kcal", "sodium": "na", "fvns_points": "lookup"})
    assert 'energy: energy_kj <- "kj", kJ; else energy_kcal <- "kcal", kcal x 4.184 to kJ' in text
    assert 'salt: sodium <- "na", g x 2.5 to salt' in text
    assert 'fvns: fvns_points <- "lookup", points as given' in text
    assert "protein: nothing" in text
