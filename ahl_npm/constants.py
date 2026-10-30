"""The fixed numbers NPM scoring needs beyond each scheme's own tables."""

# Kilojoules per kilocalorie, to score energy given in kcal: both schemes score kJ.
KJ_PER_KCAL: float = 4.184

# Grams of salt per gram of sodium, as UK labelling law defines salt (salt = sodium x 2.5, Regulation (EU) No 1169/2011,
# Annex I, as retained in UK law). A pack's salt, which NPM 2018 scores, is this; sodium chloride's mass ratio to
# sodium, 2.54, would hold only if all of a product's sodium came from table salt.
SALT_PER_SODIUM: float = 2.5

# cNPM, 70 - 2 x the NPM score, so higher is healthier: -10 to 100 under NPM 2004/05, to 110 under NPM 2018.
CNPM_OFFSET: float = 70
CNPM_SCALE: float = -2
CNPM_BOUNDS: tuple[float, float] = (0, 100)

# A value scores a threshold's point only above the threshold by more than BAND_ATOL + BAND_RTOL x the threshold, so
# float error (100 x a total / a mass, kcal x 4.184) never crosses one. These are `np.isclose`'s tolerances, which
# ahl_core_data has always snapped values to thresholds with, so scores match its.
BAND_ATOL: float = 1e-4
BAND_RTOL: float = 1e-5
