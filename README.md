# Predicting-CI-Claims-RTSI-2025
Our World in Data and the World Health Organization report that 74% of global deaths in 2019 were attributed to Noncommunicable Diseases (NCDs). Preventive care best improves NCD outcomes, while insurance provides financial protection through the Critical Illness (CI) coverage. A further step can be taken by incorporating NCD management in the administration of CI insurance through a data science approach. This research proposed a method for identifying insurance clients at risk of NCDs by estimating their probability of making a CI claim. The model consisted of outlier handling through Winsorisation, min-max Standardisation and one of five machine learning algorithms: Logistic Regression with Elastic Net (LREN), Support Vector Machine (SVM), Extreme Gradient Boosting (XGBoost), Random Forests (RF), and Artificial Neural Network (ANN). L1, L2 and implicit regularisation provided robustness to multicollinearity and the ``curse of dimensionality". The model was tuned via randomised grid search, probability calibration, and cross-validation. Experiments tested whether health insurance claims were valid predictors of critical illness by varying the dataset. Experiments also evaluated the prediction window or the ability to estimate CI risk 0, 90, 180 and 360 days beforehand. Without health claim data, models produced log skill scores of 0.074 to 0.098 and ROC-AUC scores of 0.77 to 0.79. Performance improved significantly with health claim data, with log skill scores of 0.13 to 0.39 and ROC-AUC of 0.84 to 0.91. This showed that health claims provided better predictors of CI claims. Increasing the prediction time window offered more effective interventions, but at a performance cost. Insurance providers can use the approach to assist customers with preventive NCD Management.

## Items in repo
Python Files:
- proposed_methodology.py: Python code for the proposed methodology of the research.
- create_data_partitions.py: Python code for the creation of the train-test partitions for modelling.

Notebooks:
- coverage_only_analysis.ipynb: Jupyter notebook of the analysis using critical illness coverage and client demographic information only.
- coverage_and_health_claim_analysis.ipynb: Jupyter notebook of the analysis using health claim data as indicators of client health.
- results_comparison.ipynb: Jupyter notebook comparing the results of the above experiments.
- EDA/EDA_CI_CVG.ipynb: Jupyter notebook of Exploratory Data Analysis of the critical illness coverage dataset.
- EDA/EDA_CI_CVG_HEALTH.ipynb: Jupyter notebook of Exploratory Data Analysis of the health claim dataset. 
