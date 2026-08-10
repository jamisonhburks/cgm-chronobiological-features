Chronobiologically-Informed Features from CGM Data
==================================================

Analysis code for:
Burks JH, Joe L, Kanjaria K, Monsivais C, O'Laughlin K, Smarr BL.
Chronobiologically-informed features from CGM data provide unique information for
XGBoost prediction of longer-term glycemic dysregulation in 8,000 individuals with
type-2 diabetes. PLOS Digital Health (2025). DOI: 10.1371/journal.pdig.0000815

Code repository: https://github.com/jamisonhburks/cgm-chronobiological-features


Please cite as:
Data from: Chronobiologically-Informed Features from CGM Data Provide Unique Information for XGBoost Prediction of Longer-Term Glycemic Dysregulation in 8,000 Individuals with Type-2 Diabetes (DOI: 10.6075/J0BR8SK9)

Corresponding author:
Benjamin L. Smarr (bsmarr@ucsd.edu)

Primary associated publication:
Burks et al. Chronobiologically-Informed Features from CGM Data Provide Unique Information for XGBoost Prediction of Longer-Term Glycemic Dysregulation in 8,000 Individuals with Type-2 Diabetes. PLOS Digital Health. (2025). DOI: 10.1371/journal.pdig.0000815


Description of contents:
README.txt - metadata regarding the data object's associated publication, usage code, and structure.

cgm_glycemic_prediction.ipynb - a Python Jupyter notebook that will recapitulate the figures and results for the machine learning portion of the associated publication.

src/ - a reusable Python package the notebook imports from (data.py, models.py, evaluation.py, plots.py).

data_dictionary.xlsx - a data dictionary for the processed_CGM_data_for_ML data structure.

processed_CGM_data_for_ML.parquet - a parquet file containing the tabular data used for prediction in the machine learning task. Archived separately (see Data below); not included in the code repository.


Data:
The processed dataset (processed_CGM_data_for_ML.parquet, ~331 MB) is archived at DOI 10.6075/J0BR8SK9. It exceeds GitHub's file-size limit and is not stored in the code repository. To run the notebook, download the file from the deposit and place it in the root of the repository (the notebook reads it with a relative path). The data is de-identified; ages of 90 or older are aggregated to a single value of 90 per HIPAA Safe Harbor. See data_dictionary.xlsx for a full field-by-field description.


Setup:
Analysis was performed with Python 3.11.9.
    python3 -m venv .venv
    source .venv/bin/activate        (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt


Running:
Run the notebook from the repository root (so "import src" resolves), with
processed_CGM_data_for_ML.parquet placed in that root directory:
    jupyter notebook cgm_glycemic_prediction.ipynb
Run the cells top to bottom to reproduce the ML figures and metrics.

The notebook is a thin orchestration layer over the src/ package, e.g.:
    from src import (load_dataset, split_features_label, stratified_split,
                     build_classifier, cross_validate_report)
    data = load_dataset(statistical_only=False)   # or True for the statistical-only Model 1
    X, y = split_features_label(data)
    Xtrain, Xtest, ytrain, ytest = stratified_split(X, y, random_state=42)
    metrics = cross_validate_report(build_classifier(random_state=42),
                                    Xtrain, ytrain, Xtest, ytest)


Methods:
The raw continuous glucose monitoring (CGM) data from which the tabulated features in processed_CGM_data_for_ML were obtained using the Dexcom G6 CGM data from 8000 participants.


Technical details:
All data analysis was performed in Python version 3.11.9. The following libraries were used for data analysis and statistical testing: pandas (v2.2.2) for dataframe manipulation; numpy (v1.26.4) for array manipulation, data transformations, and numerical calculations (including the multiscale complexity index); scipy.stats (v1.13.1) and statsmodels (v0.14.2) for statistical testing; matplotlib (v3.8.4) for plotting and data visualization; xgboost (v2.1.2) for classification model development; sklearn (v1.5.0) for calculation of model performance metrics.


License:
Code: MIT License (see LICENSE).
Data (the deposit at DOI 10.6075/J0BR8SK9): Creative Commons Attribution 4.0 International Public License.
