# Principal Component Analysis for Multivariate Statistical Process Control

This repository is dedicated to the development of a comprehensive Principal Component Analysis (PCA) framework specifically designed for Multivariate Statistical Process Control (MSPC). The primary goal of this framework is to provide a robust set of tools that enable users to effectively train PCA models and compute critical statistics, namely Hotelling's T^2 and the Squared Prediction Error (SPE), which are essential for anomaly detection in dynamic processes.

This project builds upon the foundational work described in the following studies:

- Ferrer, A. (2007). Multivariate statistical process control based on principal component analysis (MSPC-PCA): Some reflections and a case study in an autobody assembly process. Quality Engineering, 19(4), 311-325.

- Ferrer, A. (2014). Latent structures-based multivariate statistical process control: A paradigm shift. Quality Engineering, 26(1), 72-91.

## Features

- **PCA Model Training**: Offers a streamlined process for training PCA models tailored for MSPC, ensuring that the models capture the essential variance within the process data.
- **Computation of Key Statistics**: Implements efficient algorithms to compute Hotelling's T^2 and SPE statistics, which are pivotal for monitoring the health of the process.
- **Control Charts**: Utilizes the computed statistics to generate control charts, a fundamental component in MSPC for visualizing and detecting deviations from normal process behavior.

## Getting Started

To get started with this PCA framework for MSPC, please follow the instructions below:

1. Clone the repository to your local machine.
2. Create an environment with Python 3.10 or later.
3. Install the package and development tools with `pip install -r requirements-dev.txt`.

## Usage
After setting up the framework, you can begin training your PCA model and generating control charts.

```python
import pandas as pd
from pca_tools import PCA

# Rows are in-control reference observations; columns are measured variables.
reference = pd.read_parquet("in_control_reference.parquet")
model = PCA(n_comps=3, alpha=0.99).fit(reference)

# Phase II monitoring data must have the same names and column order.
hotelling_t2, spe, residuals, scores = model.project(new_observations)
flags = model.predict(new_observations)
```

### Modeling conventions

The implementation follows the latent-structure MSPC perspective described by
Ferrer (2014): correlated process variables are represented in a lower-dimensional
latent space, while departures from that space are monitored in the residual space.
Consequently, `PCA()` chooses a component count that preserves both a residual
subspace for SPE/DModX and the degrees of freedom needed for Phase I/II T² limits.
Set `n_comps` explicitly when process knowledge or a validated
component-selection procedure supports a different choice.

- Hotelling's T² is evaluated from PCA score variances.
- SPE (Q) and DModX are evaluated in the fitted PCA space. With standardization,
  this prevents variables with larger engineering units from dominating residual alarms.
- A full-rank model (`n_comps == n_features`) has no residual subspace. Its SPE and
  DModX limits are reported as `NaN`, and `dmodx()` raises a clear error.
- Input data must be numeric, finite, and match the fitted feature names and order.
  Impute missing values before fitting or monitoring (for example, with
  `pca_tools.pca_imputation`).

### Phase I

Phase I of the PCA framework for Multivariate Statistical Process Control (MSPC) is designed as the foundational step in establishing a robust process monitoring system. This phase encompasses the initial setup and calibration of the PCA model using historical process data, which is presumed to be reflective of the process under normal operating conditions. Here's a detailed breakdown of how Phase I works:

1. **Loading Process Data**: The first step involves importing your historical process data into the framework. This data should ideally represent the normal operating conditions of the process to serve as a baseline for anomaly detection. The framework is designed to handle various data formats and structures, making it easy to integrate with existing data collection systems.

2. **Training the PCA Model**: With the process data loaded, the next step is to train the PCA model. This involves using statistical techniques to reduce the dimensionality of the process data while retaining the most significant variations. The PCA model learns the normal behavior patterns of the process, which are crucial for identifying deviations in later stages.

3. **Computing Hotelling's T^2 and SPE Statistics**: Once the PCA model is trained, the framework computes two critical statistics: Hotelling's T^2 and the Squared Prediction Error (SPE). These statistics are derived from the PCA model and serve as indicators of process health. Hotelling's T^2 measures the variation within the model's reduced dimensionality space, while SPE focuses on the reconstruction error, indicating deviations from the model's learned behavior.

4. **Generating and Analyzing Control Charts**: The final step in Phase I involves generating control charts based on the computed Hotelling's T^2 and SPE statistics. These charts visualize the statistical health of the process over time, providing a clear and intuitive means to monitor for any anomalies. By setting control limits on these charts, users can easily identify when the process deviates from its normal operating conditions, signaling potential issues that require further investigation.

Once the anomalies are correctly identified, we can optimize the model (i.e. removed the largest anomalies from the training set), and verify its ability to detect outliers on a test set.

### Phase II

Phase II of the PCA framework for Multivariate Statistical Process Control (MSPC) focuses on the real-time monitoring and control of the process using the PCA model calibrated in Phase I. This phase is crucial for the dynamic application of the framework to detect and diagnose anomalies as they occur in the process, aligning with the repository's aim to provide robust tools for anomaly detection in dynamic processes. Here's an overview of how Phase II operates:

1. **Application of the PCA Model**: Apply the PCA model, trained during Phase I, to the new real-time or test data. This involves projecting the new data onto the PCA model to compute the current Hotelling's T^2 and SPE statistics, which are essential for identifying deviations from the process's normal operating conditions.

2. **Anomaly Detection**: Utilize the computed Hotelling's T^2 and SPE statistics to detect anomalies in real-time. Anomalies are identified when these statistics exceed the control limits established in Phase I, indicating a deviation from the normal process behavior.

3. **Diagnostic Analysis**: Once an anomaly is detected, the framework facilitates a diagnostic analysis to identify the source of the deviation. This can be done by using the methods `spe_contribution_plot` and `hotelling_t2_contribution_plot`, depending on the control chart that has detected the anomaly. Both this functions can be found at `pca_tools/utils.py`

4. **Model Optimization and Validation**: In cases where the process undergoes significant changes, the PCA model may be re-optimized using new data that includes the detected anomalies. This ensures that the model remains accurate and effective in detecting outliers. The optimized model is then validated to ensure its efficacy in real-time anomaly detection.

After the optimization carried out during Phase I and the calibration and diagnosis on Phase II, the model can be saved as a `pickle` object and used in a production environment, where anomalies can be detected in real-time. The resulting object is compatible with Scikit-learns `Pipeline` object.

## Definition of the outliers

- **Hotelling's T^2**: The Hotelling's T^2 chart evaluates whether the projection of an observation onto the hyperplane, as defined by the latent subspace, falls within the boundaries set by the reference (in-control) data. Consequently, when the value of this statistic surpasses the control limits, it signifies that the observation exhibits unusually extreme values across some or all of its K measured variables. This occurs despite the observation adhering to the correlation structure among the model's variables. Such observations are identified as abnormal outliers within the PCA model, indicating they are extreme or severe outliers.

- **SPE (Squared Prediction Error)**: The SPE chart is designed to measure the distance, or noise variation, of an observation from the latent hyperplane, ensuring it remains within the predefined control limits. When SPE chart values exceed these limits, it indicates that the observation deviates from the behavior of the in-control data used to construct the model. Specifically, this deviation manifests as a disruption in the correlation structure established by the model. The SPE chart is adept at identifying the emergence of any novel events that cause the process to diverge from the hyperplane defined by the reference model. Observations identified through this method are classified as outliers external to the model, denoting them as alien or moderate outliers.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
