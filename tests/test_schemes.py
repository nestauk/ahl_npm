from importlib import resources
from pathlib import Path

import pytest

from ahl_npm import Component, band, get_scheme, load_scheme, score


def test_score_ranges_match_the_guidance() -> None:
    """NPM 2004/05 scores range from -15 to 40, and NPM 2018 scores from -20 as fibre can score up to 10 points."""
    assert get_scheme("2004").score_range == (-15, 40)
    assert get_scheme("2018").score_range == (-20, 40)


def test_sodium_thresholds_are_exact_in_grams() -> None:
    """The guidance's mg thresholds convert to exactly the same floats as when written in g."""
    in_grams = (0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72, 0.81, 0.9)
    assert get_scheme("2004").component("sodium").thresholds == in_grams


def test_npm_2018_protein_scores_four_points_above_6_8() -> None:
    """The technical guidance gives 4 points above 6.8 g/100g."""
    assert band("2018", "protein", [6.8, 6.81, 8.5, 8.51]).tolist() == [3, 4, 4, 5]


def test_fvn_points_jump_to_five() -> None:
    """Fruit, vegetables and nuts score 1 above 40%, 2 above 60% and 5 above 80%."""
    assert band("2004", "fvn", [40, 40.5, 60, 61, 80, 81]).tolist() == [0, 1, 1, 2, 2, 5]


def test_unknown_schemes_and_components_raise() -> None:
    """An unknown scheme or component raises an error rather than falling back to a default."""
    with pytest.raises(ValueError, match="2021"):
        get_scheme("2021")
    with pytest.raises(KeyError, match="free_sugars"):
        get_scheme("2004").component("free_sugars")


def test_own_schemes_load_and_score(tmp_path: Path) -> None:
    """A custom scheme, here NPM 2018 with salt's thresholds halved, can be used anywhere a scheme's name can."""
    text = (resources.files("ahl_npm") / "schemes" / "npm_2018.yaml").read_text()
    halved = text.replace('name: "2018"', 'name: "2018 half salt"').replace(
        "[0.2, 0.5, 0.7, 0.9, 1.1, 1.4, 1.6, 1.8, 2.0, 2.3]", "[0.1, 0.25, 0.35, 0.45, 0.55, 0.7, 0.8, 0.9, 1.0, 1.15]"
    )
    (tmp_path / "half_salt.yaml").write_text(halved)
    scheme = load_scheme(tmp_path / "half_salt.yaml")
    assert scheme.name == "2018 half salt"
    assert band(scheme, "salt", [0.5, 1.2]).tolist() == [4, 10]
    panel = {"energy_kj": 0, "satfat": 0, "free_sugars": 0, "salt": 1.2, "protein": 0, "fibre": 0, "fvns_pct": 0}
    assert score(scheme, **panel)["npm"].item() == 10
    assert score("2018", **panel)["npm"].item() == 5


def test_schemes_only_take_components_inputs_exist_for(tmp_path: Path) -> None:
    """An unknown component, e.g. total fat, raises an error when the scheme is loaded."""
    text = (resources.files("ahl_npm") / "schemes" / "npm_2004.yaml").read_text()
    (tmp_path / "fat.yaml").write_text(text.replace("  satfat:", "  fat:"))
    with pytest.raises(ValueError, match=r"components \['fat'\]"):
        load_scheme(tmp_path / "fat.yaml")


def test_components_must_be_step_functions() -> None:
    """Thresholds must increase, with points for each."""
    with pytest.raises(ValueError, match="increase"):
        Component("x", "A", (2.0, 1.0), (1, 2))
    with pytest.raises(ValueError, match="points"):
        Component("x", "A", (1.0, 2.0), (1,))
