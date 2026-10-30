"""Constants used in NPM scoring that are not part of a scheme's thresholds."""

# Kilojoules per kilocalorie, used to convert energy given in kcal, as both schemes score energy in kJ.
KJ_PER_KCAL: float = 4.184

# Grams of salt per gram of sodium, as UK labelling law defines salt (salt = sodium x 2.5, Regulation (EU) No 1169/2011,
# Annex I, as retained in UK law).
SALT_PER_SODIUM: float = 2.5

# cNPM is 70 - 2 x the NPM score, s.t. higher is healthier. It runs from -10 to 100 under NPM 2004/05, and up to 110
# under NPM 2018. CNPM_BOUNDS are the limits used when clipping.
CNPM_OFFSET: float = 70
CNPM_SCALE: float = -2
CNPM_BOUNDS: tuple[float, float] = (0, 100)

# A value scores a threshold's point only above the threshold by more than BAND_ATOL + BAND_RTOL x the threshold,
# this prevents floating point error from causing a value to be scored at a different threshold.
BAND_ATOL: float = 1e-4
BAND_RTOL: float = 1e-5
