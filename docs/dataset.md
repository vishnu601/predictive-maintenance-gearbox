# Dataset Provenance & Ingestion Specifications

## 1. Primary Dataset Overview

The platform uses the authoritative open-access **Gearbox Fault Diagnosis Data**, cataloged on **US Data.gov** and hosted by the **US Department of Energy (DOE) OpenEI**:
- **Data.gov Catalog URL:** [https://catalog.data.gov/dataset/gearbox-fault-diagnosis-data](https://catalog.data.gov/dataset/gearbox-fault-diagnosis-data)
- **Direct Download Archive:** [https://data.openei.org/files/623/gearboxdata.zip](https://data.openei.org/files/623/gearboxdata.zip)
- **Authoritative Publisher:** Prestige Institute of Engineering Management and Research (PIEMR) & DOE.
- **Academic Reference:** *"Fault detection of a two-stage gearbox with artificially induced gear crack using vibration monitoring"*.

---

## 2. Experimental Rig & Sensor Configuration

The dataset was collected using an industrial-grade **SpectraQuest Gearbox Diagnostics Simulator (GDS)**:
- **Mechanical Structure:** Two-stage spur gearbox with induction motor drive and magnetic brake load simulator.
- **Shaft Speed:** Constant 30 Hz nominal rotational speed ($\approx 1,800$ RPM).
- **Vibration Sensors:** 4 piezoelectric accelerometers mounted in orthogonal directions:
  - Channel 1: Transverse / Radial Vibration
  - Channel 2: Axial Vibration
  - Channel 3: Vertical Vibration
  - Channel 4: Torsional / Casing Vibration
- **Sampling Frequency:** Nominal 5,000 Hz.

---

## 3. Recording Structure & Ground Truth Labels

The archive `gearboxdata.zip` contains exactly 20 distinct recording files partitioned by loading condition (0% to 90% in 10% increments):

| Directory | Filename | Operating Condition | Load (%) | Channel Count | Samples per File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Healthy Data/` | `h30hz0.txt` to `h30hz90.txt` | Healthy Normal Spur Gear | 0% – 90% | 4 | ~88,800 |
| `BrokenTooth Data/` | `b30hz0.txt` to `b30hz90.txt` | Artificially Induced Gear Crack / Tooth Damage | 0% – 90% | 4 | ~88,800 |

Total Raw Data Volume: **1,776,680 four-channel accelerometer observations**.

---

## 4. Ground Truth Mapping & Non-Fabrication Policy

Per strict engineering standards, we report only the true physical conditions evaluated in this experimental setup:
- **Supported Class 1:** `Healthy` (normal gear dynamics)
- **Supported Class 2:** `BrokenTooth` (gear tooth crack / broken tooth defect)
- **Unsupported Classes:** Bearing defects, shaft misalignment, mechanical looseness, and motor electrical faults. These classes are tracked as `UNAVAILABLE_FOR_DATASET` in the platform and are never fabricated or simulated as real experimental data.
