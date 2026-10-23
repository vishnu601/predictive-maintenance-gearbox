# Evaluation Protocol & Benchmark Methodology

## 1. Preventing Data Leakage in Vibration Monitoring

In rotating mechanical systems, vibration signals sampled closely in time exhibit high serial autocorrelation. If sliding windows cut from a single recording are randomly shuffled into training and test partitions, the model effectively memorizes neighboring segments of the exact same physical run. This produces artificially inflated accuracy ($>99.9\%$) that collapses when deployed on an unseen machine.

### Our Leakage Prevention Protocol
1. **Group Hierarchy:** Each of the 20 experimental files represents an independent physical run with unique loading (0% to 90%).
2. **Disjoint Partitioning:** All windows derived from a recording file belong to *exactly one* partition:
   - Training Partition: 65% of recording groups
   - Validation Partition: 15% of recording groups (used for early stopping & hyperparameter verification)
   - Held-Out Test Partition: 20% of recording groups (completely unseen until final evaluation)
3. **Automated Assertion:** `verify_no_group_leakage()` runs in both unit tests and dataset preparation, throwing a `ValueError` if any recording ID appears in more than one split.

---

## 2. Evaluation Metrics

### Primary Metric: Macro-Averaged F1 Score
$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$
Where $C$ is the number of supported classes. Macro-F1 gives equal weight to both healthy operation and rare mechanical failures, penalizing models that merely predict the majority class.

### Secondary Metrics
- **Balanced Accuracy:** The unweighted mean of recall obtained on each class.
- **Confusion Matrix:** True Positives, False Positives, True Negatives, and False Negatives, along with normalized percentages.
- **Inference Latency:** Measured over 100 consecutive window predictions on CPU after 5 warmup cycles, reported in milliseconds (mean and p95).
- **Model Footprint:** Serialized disk size of the artifact in kilobytes (KB).
