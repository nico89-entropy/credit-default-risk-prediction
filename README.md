# Credit Default Risk Prediction

An end-to-end machine learning project for estimating the risk that a credit-card customer will default on the following month's payment.

The project combines exploratory data analysis, behavioral feature engineering, interpretable classification, class-imbalance-aware evaluation, and cost-sensitive decision-threshold selection.

## Business Problem

Credit default is an important source of financial risk for lending institutions. A useful model should not only classify customers, but also estimate default probabilities and support decisions in which different errors have different consequences:

- **False negative:** a customer predicted as low-risk later defaults.
- **False positive:** a reliable customer is incorrectly flagged as high-risk.

Because false negatives may be more costly, this project evaluates the trade-off between recall, precision, and an illustrative classification cost rather than optimizing accuracy alone.

## Dataset

The project uses the [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset from the UCI Machine Learning Repository.

The dataset contains 30,000 customer records and information about:

- Credit limits
- Demographic characteristics
- Six months of payment status
- Monthly bill amounts
- Monthly payment amounts
- Default status for the following month

The target variable is binary:

- `0`: no default
- `1`: default

Approximately 22.12% of the observations belong to the default class, producing a moderately imbalanced classification problem.

## Project Workflow

1. Download and cache the dataset locally.
2. Assess missing values, duplicates, data types, and categorical codes.
3. Clean undocumented education and marital-status categories.
4. Explore default rates across customer and financial characteristics.
5. Engineer behavioral features from six months of payment history.
6. Create stratified training, validation, and test sets.
7. Build a preprocessing pipeline for numerical and categorical variables.
8. Establish a majority-class baseline.
9. Compare logistic regression and random forest models.
10. Select a decision threshold using an illustrative cost function.
11. Evaluate the selected model on an untouched test set.
12. Analyze feature importance and model limitations.

## Feature Engineering

The notebook creates behavioral summaries including:

- `months_with_delay`: number of months with a recorded payment delay
- `maximum_delay`: most severe delay observed during the six-month period
- `average_bill_amount`: average monthly bill amount
- `average_payment_amount`: average monthly payment amount
- `total_bill_amount`: total billed amount over six months
- `total_payment_amount`: total amount paid over six months

These features summarize the persistence and severity of previous payment difficulties.

## Models

The following models are included:

- `DummyClassifier` as a majority-class baseline
- Logistic regression with balanced class weights
- Random forest with balanced class weights

Logistic regression provides an interpretable linear benchmark, while random forest can capture nonlinear relationships and interactions.

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision
- Confusion matrix
- ROC curve
- Precision-recall curve
- Calibration curve

Model comparison and threshold selection use the validation set. The test set remains untouched until the final evaluation.

## Cost-Sensitive Threshold Selection

The default probability threshold of `0.50` is compared with a validation-selected threshold. For demonstration purposes, a false negative is assigned five times the cost of a false positive:

```text
Estimated cost = 1 × false positives + 5 × false negatives
```

These weights are illustrative. A real financial application would require institution-specific monetary costs, risk policies, and regulatory review.

## Results

Final values should be copied from the completed notebook run before presenting the project as final.

| Model / threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Dummy baseline | Pending | Pending | Pending | Pending | Pending | Pending |
| Selected model, threshold 0.50 | Pending | Pending | Pending | Pending | Pending | Pending |
| Selected model, optimized threshold | Pending | Pending | Pending | Pending | Pending | Pending |

## Project Structure

```text
credit-default-risk-prediction/
├── data/                                  # Local dataset cache; ignored by Git
├── images/                                # Selected figures for the README
├── models/                                # Optional serialized models; ignored by Git
├── notebooks/
│   └── credit_default_risk_prediction.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/nico89-entropy/credit-default-risk-prediction.git
cd credit-default-risk-prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/credit_default_risk_prediction.ipynb
```

The dataset is downloaded from UCI during the first execution and cached under `data/` for subsequent runs.

## Responsible Use

This project is an educational prototype, not an automated credit-decision system. Demographic variables such as sex can introduce fairness concerns and should not be used automatically to approve or reject credit applications. Predictions require human oversight, fairness assessment, appropriate governance, and compliance with applicable regulations.

## Limitations

- The data describes Taiwanese credit-card customers and may not generalize to other countries, institutions, or time periods.
- The cost assumptions used for threshold selection are illustrative.
- Feature importance indicates predictive association, not causality.
- Customer behavior and economic conditions can change over time.
- A production model would require stronger validation, monitoring, security, and fairness analysis.

## Future Improvements

- Add stratified cross-validation and hyperparameter optimization.
- Compare gradient-boosting models.
- Measure performance and calibration across demographic groups.
- Evaluate models with and without sensitive attributes.
- Replace illustrative costs with institution-specific financial estimates.
- Monitor data drift and performance degradation over time.

## Author

**Juan Nicolas Moreno**

- GitHub: [nico89-entropy](https://github.com/nico89-entropy)

## Acknowledgment

Dataset: I-Cheng Yeh and Che-hui Lien, *The Comparisons of Data Mining Techniques for the Predictive Accuracy of Probability of Default of Credit Card Clients*, Expert Systems with Applications, 2009.
