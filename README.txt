Please cite as:
Data from: Chronobiologically-Informed Features from CGM Data Provide Unique Information for XGBoost Prediction of Longer-Term Glycemic Dysregulation in 8,000 Individuals with Type-2 Diabetes (DOI: 10.6075/J0BR8SK9)

Corresponding author:
Benjamin L. Smarr (bsmarr@ucsd.edu)

Primary associated publication:
Burks et al. Chronobiologically-Informed Features from CGM Data Provide Unique Information for XGBoost Prediction of Longer-Term Glycemic Dysregulation in 8,000 Individuals with Type-2 Diabetes. PLOS Digital Health. (2025) 

Description of contents:
README.txt - metadata regarding the data object's associated publication, usage code, and structure.

ComplexityML.ipynb - a Python Jupyter notebook that will recapitulate the figures and results for the machine learning portion of the associated publication.

processed_CGM_data_for_ML.parquet - a parquet file containing the tabular data used for prediction in the machine learning task.

ProcessedCGM_dataDict.xlsx - a data dictionary for the processed_CGM_data_for_ML data structure.

Methods:
The raw continuous glucose monitoring (CGM) data from which the tabulated features in processed_CGM_data_for_ML were obtained using the Dexcom G6 CGM data from 8000 participants.

Technical details:
All data analysis was performed in Python version 3.11.9. The following libraries were used for data analysis and statistical testing: pandas (v2.2.2) for dataframe manipulation; numpy (v1.26.4) for array manipulation, data transformations, and numerical calculations (including the multiscale complexity index); scipy.stats (v1.13.1) and statsmodels (v0.14.2) for statistical testing; matplotlib (v3.8.4) for plotting and data visualization; xgboost (v2.1.2) for classification model development; sklearn (v1.5.0) for calculation of model performance metrics.

License:
Creative Commons Attribution 4.0 International Public License.
