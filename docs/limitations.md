# Limitations, Scope Boundaries & Honest Engineering Appraisal

This document clarifies what the platform has demonstrated, what remains future engineering work, and where caution must be exercised.

---

## 1. Fault Diagnosis vs. Remaining Useful Life (RUL)

- **What Is Implemented:**
  - **Early Fault Detection:** Distinguishing normal operating vibration patterns from developing mechanical anomalies.
  - **Fault Classification:** Categorizing vibration signals into healthy gear meshing or gear tooth damage/cracking.
  - **Anomaly Scoring:** Quantifying deviation from healthy baselines using Isolation Forests.
- **What Is NOT Implemented (RUL):**
  - The platform **does not estimate Remaining Useful Life (RUL)** in hours or cycles.
  - *Engineering Reason:* RUL estimation requires run-to-failure degradation datasets with monotonic wear progression. The OpenEI dataset contains discrete static damage states (healthy vs. cracked), not continuous multi-month run-to-failure telemetry. Any RUL claim on this dataset would be mathematically ungrounded and misleading.

---

## 2. Laboratory Data vs. Factory Production Environment

- The models were trained and verified on recordings from a **SpectraQuest Gearbox Diagnostics Simulator** under steady laboratory loads (0% to 90%).
- Real industrial gearboxes (e.g., in wind turbines, mining conveyors, or marine propulsion) experience:
  - Complex multi-stage planetary gear meshes
  - Variable and shock loading conditions
  - Background acoustic noise and structural resonance from adjacent machinery
- **Deployment Recommendation:** Before deploying in a factory environment, record at least 48 hours of healthy baseline data on the specific target asset to recalibrate the Isolation Forest anomaly threshold and verify sensor mounting.

---

## 3. Decision Support vs. Safety-Critical Automation

- The platform produces **cautious maintenance recommendations** intended as diagnostic decision aids for qualified reliability engineers.
- It is **not** an automated safety-instrumented system (SIS) and must never be configured to trigger emergency shutdowns or interlocks without human review.

---

## 4. Next Roadmap Improvements
1. **Multi-Channel Synchronous Processing:** Expanding the 1D CNN to a 4-channel 2D/1D multi-sensor input tensor combining orthogonal accelerometer directions.
2. **CWRU Bearing Data Integration:** Adding specialized bearing defect classification (inner raceway, outer raceway, ball fault) when separate bearing accelerometer streams are active.
3. **Edge Microcontroller Deployment:** Exporting the trained 1D CNN via ONNX/TensorFlow Lite Micro for direct inference on ARM Cortex-M microcontrollers.
