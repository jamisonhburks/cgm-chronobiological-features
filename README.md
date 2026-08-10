# Chronobiologically-Informed Features from CGM Data

Analysis code for:

> Burks JH, Joe L, Kanjaria K, Monsivais C, O'Laughlin K, Smarr BL.
> **Chronobiologically-informed features from CGM data provide unique information for
> XGBoost prediction of longer-term glycemic dysregulation in 8,000 individuals with
> type-2 diabetes.** *PLOS Digital Health* (2025).
> [doi:10.1371/journal.pdig.0000815](https://doi.org/10.1371/journal.pdig.0000815)

This repository contains the machine-learning notebook that reproduces the figures and
results for the prediction task in the paper. The processed dataset is archived
separately (see **Data** below). The full open-access article is available on
[PLOS Digital Health](https://doi.org/10.1371/journal.pdig.0000815).

## Repository contents

| File | Description |
|------|-------------|
| `cgm_glycemic_prediction.ipynb` | Jupyter notebook that recapitulates the figures and results for the ML portion of the paper (XGBoost prediction of longer-term glycemic dysregulation). |
| `src/` | Reusable analysis package the notebook imports from (see below). |
| `data_dictionary.xlsx` | Data dictionary for `processed_CGM_data_for_ML.parquet` (52 fields). |
| `requirements.txt` | Python dependencies and pinned versions. |

### Reusable code (`src/`)

The notebook is a thin orchestration layer over a small importable package so the same
logic can be reused in future analyses:

```
src/
├── data.py         # loading, feature-group definitions, feature_matrix, stratified_split
├── models.py       # build_classifier (single source of hyperparameters), fit_predict
├── evaluation.py   # per_class_roc_auc, cross_validate_report, calibration_points
└── plots.py        # calibration curves, feature-importance comparison, PCA biplot
```

```python
from src import load_dataset, split_features_label, stratified_split, build_classifier, cross_validate_report

data = load_dataset(statistical_only=False)          # or True for the statistical-only Model 1
X, y = split_features_label(data)
Xtrain, Xtest, ytrain, ytest = stratified_split(X, y, random_state=42)
metrics = cross_validate_report(build_classifier(random_state=42), Xtrain, ytrain, Xtest, ytest)
```

Run the notebook from the repository root so `import src` resolves.

## Data

The processed dataset (`processed_CGM_data_for_ML.parquet`, ~331 MB, 1,457,219 rows ×
52 columns derived from Dexcom G6 CGM data for 8,000 participants) is **not** stored in
this repository because it exceeds GitHub's file-size limit. It is archived at:

- **DOI:** [10.6075/J0BR8SK9](https://doi.org/10.6075/J0BR8SK9)

To run the notebook, download `processed_CGM_data_for_ML.parquet` from the deposit and
place it in the **root of this repository** (the notebook reads it with a relative path).

The data is de-identified. Ages of 90 or older are aggregated to a single value of `90`
per HIPAA Safe Harbor. See `data_dictionary.xlsx` for a full field-by-field
description.

The raw CGM data from which these tabulated features were derived are Dexcom G6
continuous glucose monitoring recordings from 8,000 participants.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Analysis was performed with **Python 3.11.9**. Key library versions: pandas 2.2.2,
numpy 1.26.4, scipy 1.13.1, statsmodels 0.14.2, matplotlib 3.8.4, xgboost 2.1.2,
scikit-learn 1.5.0.

## Running

```bash
jupyter notebook cgm_glycemic_prediction.ipynb
```

With `processed_CGM_data_for_ML.parquet` in the repository root, run the cells top to
bottom to reproduce the ML figures and metrics.

## Citation

Please cite both the article and the dataset. See [`CITATION.cff`](CITATION.cff) for
machine-readable metadata.

- **Article:** Burks et al., *PLOS Digital Health* (2025), doi:10.1371/journal.pdig.0000815
- **Data:** *Data from: Chronobiologically-Informed Features from CGM Data Provide Unique
  Information for XGBoost Prediction of Longer-Term Glycemic Dysregulation in 8,000
  Individuals with Type-2 Diabetes*, doi:10.6075/J0BR8SK9

## License

- **Code** (this repository): [MIT](LICENSE)
- **Data** (the deposit at DOI 10.6075/J0BR8SK9): Creative Commons Attribution 4.0
  International (CC-BY 4.0)

## Contact

Corresponding author: Benjamin L. Smarr — bsmarr@ucsd.edu
