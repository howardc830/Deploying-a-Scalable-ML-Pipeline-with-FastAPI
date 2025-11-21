# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details ##
This is a binary classification model that predicts whether an individual's annual income exceeds $50,000.

- **Model**: Random Forest Classifier (n_estimators=100-200, random_state=28)
- **Features**: 14 census features (age, workclass, education, marital-status, occupation, relationship, race, sex, native-country, etc.)
- **Preprocessing**: One-Hot Encoding for categorical features + LabelBinarizer for the target.
- **Creator**: Christopher Howard
- **Date**: November 2025


## Intended Use ##
Used to demonstrate a complete end-to-end ML pipeline, including training, evaluation, slice-based fairness analysis, model persistence, and API deployment with FastAPI.

## Training Data

- **Dataset**: Adult Census Income dataset ("census.csv" - clearned version)
- **Total Records**: ~32,561 (before split)
- **Train/Test Split**: 80% train, 20% test (stratified on salary)
- **Positive Class (>50k): ~24% of data (class imbalance present)

## Evaluation Data
Held-out test set (20% of original data, ~6,513 records), never seen during training.

## Metrics
The model was evaluated using the following metrics on the test set:

Precision: 0.7960
Recall: 0.5651
F1-score: 0.6609

Additionally, performance was computed on every unique value of all 8 categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country). The full results are stored in the 'slice_output.txt'

Key observations from slice analysis:
- Excellent performance on high education groups (Doctorate, Prof-school) and executive roles.
- Lower recall on underrepresented occupations (e.g., Other-service, Handlers-cleaners).
- No catastrophic bias detected across race or sex, though minor disparities exist as expected with imbalanced data.

## Ethical Considerations
- The dataset contains sensitive attributes (race, sex, native-country). Slice metrics were explicitly calculated to monitor potential disparate impact.
- Current performance differences across groups are moderate and largely explained by class imbalance and feature disribution rather than model bias.
- In a production system, ongoing monitoring and fairness constraints would be recommended.

## Caveats and Recommendations
- The model may underperform on very rare categories (e.g., certain native countries with <10 examples).
- Class imbalance favors the <=50k class; techniques like oversampling, class weighting, or calibrated thresholds could improve recall if needed.
- This model is suitable for demonstration and research use. For real-world deployment, additional fairness audits, explainability, and drift monitoring should be implemented.

Model artifacts ('model.pkl', 'encoder.pkl', 'lb.pkl') and slice analysis ('slice_output.txt') are versioned and ready for API serving. 
