"""Settings shared by HARK's tests."""

import matplotlib

# Draw with matplotlib's non-interactive Agg backend in every test, so that no
# test needs a display or Tk. GitHub's Windows runners sometimes fail to load Tk
# ("Can't find a usable tk.tcl"), which failed tests that only make figures.
matplotlib.use("Agg")
