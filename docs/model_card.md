# Model Card: Gearbox Vibration Condition Classifiers

## 1. Model Details
- **Architecture Family:**
  1. **Random Forest Classifier:** Scikit-learn ensemble of 150 decision trees consuming 20 statistical time/frequency features.
  2. **XGBoost Classifier:** Gradient-boosted decision trees with depth regularization and validation-guided early stopping.
  3. **1D Deep Convolutional Neural Network (PyTorch):** 3 Conv1d blocks ($15\times 1, 7\times 1, 5\times 1$ receptive fields), Batch Normalization, Dropout ($p=0.3$), and Adaptive Average Pooling directly processing normalized raw waveforms.
  4. **Isolation Forest Anomaly Detector:** Unsupervised tree ensemble trained exclusively on healthy baseline recordings to quantify statistical anomaly deviations.
- **Input Dimension:**
  - Tabular models: 20 extracted vibration features (RMS, Peak-to-Peak, Kurtosis, Dominant Frequency, Spectral Energy, Spectral Entropy, etc.).
  - 1D CNN: 1D vibration acceleration tensor $(B \times 1 \times 2048)$.
- **Output:** Categorical condition prediction (`Healthy` vs. `BrokenTooth`), calibrated probability estimates, and continuous anomaly score.

---

## 2. Intended Use
- **Primary Use Case:** Real-time and offline condition monitoring for rotating gearboxes to detect gear tooth cracking or abnormal vibration before structural failure occurs.
- **Users:** Mechanical condition-monitoring technicians, reliability engineers, and predictive maintenance operators.
- **Out-of-Scope / Prohibited Uses:**
  - Automatic emergency shutdown of human-occupied machinery without human oversight.
  - Estimation of Remaining Useful Life (RUL) without empirical run-to-failure fatigue data.
  - Deployment on machines with radically different gear ratios or speeds without recalibration.

---

## 3. Training & Evaluation Protocol
- **Data Source:** US Data.gov / OpenEI Gearbox Fault Diagnosis Dataset (SpectraQuest GDS).
- **Leakage Prevention:** Group-aware splitting (`GroupShuffleSplit`) partitioned by physical recording run. No window from a test recording ever enters the training or validation splits.
- **Primary Metric:** Macro-Averaged F1 Score ($\text{Macro-F1}$), treating both healthy and faulty conditions with equal importance regardless of class balance.
- **Secondary Metrics:** Balanced Accuracy, Per-Class Recall, and Inference Latency (ms).

---

## 4. Caveats & Ethical Considerations
- High accuracy on a laboratory test simulator does not guarantee equal performance in factory environments with acoustic echoes, structural resonances, or fluctuating electrical grid frequencies.
- Confidence scores reflect relative softmax or leaf certainty under the training distribution and must not be misinterpreted as exact physical failure probabilities.
