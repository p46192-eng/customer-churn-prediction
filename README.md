# AI-Based Customer Churn Prediction & Managerial Decision Support

This project develops a customer churn prediction model and a Streamlit decision-support tool for the Business Analytics course.

## Business problem
Customer churn can reduce recurring revenue and increase customer-acquisition costs. The objective is to use historical customer behaviour to estimate the probability that a customer will churn, so managers can prioritise customers for further review and retention action.

## Dataset
The project uses the **Iranian Churn** dataset from the UCI Machine Learning Repository (Dataset ID 563).

- 3,150 customer records
- 13 predictor features
- Binary churn outcome
- No missing values
- Predictor information is aggregated from the first 9 months; churn is the customer's state at the end of month 12, with a three-month planning gap.

Source: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset  
DOI: https://doi.org/10.24432/C5JW3Z  
License: CC BY 4.0

## Model
A Logistic Regression classifier is used because the assignment requires a classification model and the output probability is directly useful for managerial risk screening.

The workflow is:
1. Fetch and validate the UCI dataset.
2. Separate predictors and churn target.
3. Split data into training and test sets using stratification.
4. Standardise numerical variables.
5. Train Logistic Regression with balanced class weights.
6. Evaluate using Accuracy, Precision, Recall, F1-score and ROC-AUC.
7. Generate a confusion matrix.
8. Use the fitted pipeline in the Streamlit decision-support interface.

## Important interpretation
The model estimates the probability of the **documented churn outcome**. It is a decision-support signal, not a guarantee that an individual customer will leave. Managers should consider customer context and their own judgement before taking action.

## Streamlit app
The application accepts customer-level inputs and returns:
- estimated churn probability
- risk category
- managerial interpretation
- model performance information

The risk category is a communication aid:
- High risk: probability >= 70%
- Medium risk: probability 40% to <70%
- Low risk: probability <40%

These thresholds are not claims about actual business loss or customer behaviour; they are used to make the model output easier for non-technical users to interpret.

## Run locally
Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The application retrieves the dataset through the UCI Machine Learning Repository using the `ucimlrepo` package, so the CSV does not need to be manually copied into the repository.

## Citation
Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z
