# Raw Data Directory

This directory contains raw experimental recordings downloaded from US Data.gov / OpenEI:
- `gearboxdata.zip`: Downloaded 36.7 MB archive.
- `gearbox_extracted/`: Unpacked directory containing:
  - `Healthy Data/`: 10 files (`h30hz0.txt` to `h30hz90.txt`), corresponding to loads 0% to 90%.
  - `BrokenTooth Data/`: 10 files (`b30hz0.txt` to `b30hz90.txt`), corresponding to loads 0% to 90%.

Each file is tab-separated, containing 4 accelerometer sensor channels placed in 4 orthogonal directions on the SpectraQuest Gearbox Diagnostics Simulator.
Nominal sampling rate: 5,000 Hz. Nominal shaft rotational speed: 30 Hz (~1,800 RPM).
