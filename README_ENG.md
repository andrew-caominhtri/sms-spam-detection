# SMS Spam Detection – Detecting Spam SMS Messages Using Machine Learning

## 1. Project Overview

**Project title:** SMS Spam Detection – **A Machine Learning-Based Application for Detecting Spam SMS Messages**.

This project classifies SMS messages into two categories:

- **Ham:** Legitimate, non-spam messages.
- **Spam:** Unwanted messages, which may contain advertising, promotions, or fraudulent content.

The system uses **TF-IDF** for text representation and compares two classifiers: **Multinomial Naive Bayes** and **Logistic Regression**. The machine learning experiment is documented in a **Jupyter Notebook**, while a **Streamlit** web application allows users to enter messages and view predictions interactively.

**Scope:** The current models are trained and evaluated on **English-language SMS messages**. Their performance on Vietnamese messages has not been validated.

## 2. Dataset

- **Dataset:** SMS Spam Collection.
- **Source:** UCI Machine Learning Repository – [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
- **Dataset file:** `data/sms+spam+collection/SMSSpamCollection` (tab-separated `label` and `message` fields).
- **Original dataset:** 5,574 messages: 4,827 Ham and 747 Spam.
- **After removing duplicate messages:** 5,171 messages: 4,518 Ham and 653 Spam.
- **Data split:** 80% training (4,136 messages) and 20% testing (1,035 messages), stratified by class label, with `random_state=42`.

The ZIP includes the English SMS dataset used by the project.

## 3. Machine Learning Workflow

1. Load the SMS dataset and remove duplicate records.
2. Split the cleaned data into training and test sets (80/20), preserving the approximate class distribution.
3. Extract **TF-IDF** features with unigrams and bigrams (`ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`). The vectorizer is fitted **only on the training set**.
4. Train two pipelines: **TF-IDF + Multinomial Naive Bayes** and **TF-IDF + Logistic Regression**.
5. Evaluate both models on the held-out test set using Accuracy, Precision, Recall, F1-score, and confusion matrices. **Spam is treated as the positive class.**
6. Save the trained pipelines as `.joblib` files, evaluation metrics as `.csv`, and charts as `.png`. The Streamlit app loads these artifacts.

## 4. Project Structure

```text
sms-spam-detection-converted/
├── SMS_Spam_Detection.ipynb         # Full ML experiment: data, TF-IDF, training, evaluation, model export
├── app.py                           # Streamlit interface: Detection, Performance, About
├── requirements.txt                 # Python dependencies and pinned versions
├── README.md                        # Project overview, setup instructions, and results
├── README_ENG.md                                       # English documentation
├── .gitignore                       # Files and folders excluded from Git tracking
├── data/
│   └── sms+spam+collection/
│       ├── SMSSpamCollection        # English SMS dataset used in the experiment
│       └── readme                   # Original dataset documentation
├── models/
│   ├── naive_bayes_pipeline.joblib  # Trained TF-IDF + Multinomial Naive Bayes pipeline
│   └── logistic_regression_pipeline.joblib  # Trained TF-IDF + Logistic Regression pipeline
└── outputs/
    ├── metrics.csv                  # Evaluation metrics for both models
    ├── threshold_metrics.csv                           # Test-set threshold comparison
    ├── confusion_matrix_naive_bayes.png        # Naive Bayes confusion matrix
    ├── confusion_matrix_logistic_regression.png # Logistic Regression confusion matrix
    └── model_comparison.png         # Model performance comparison chart
```

## 5. Environment Setup

**Requirements:** Python 3.11, Windows PowerShell, and the project source files. Open a terminal in the **project root** (`sms-spam-detection-converted`).

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install project dependencies
python -m pip install -r requirements.txt

# Check installed dependency compatibility
python -m pip check
```

To **open and run the Jupyter Notebook**, install the following additional packages (not required if you only want to use the Streamlit app):

```powershell
python -m pip install notebook ipykernel
```

You can open the Notebook in VS Code with the Jupyter extension and select the Python interpreter from `.venv`.

## 6. Running the Streamlit Application

From the project root, after activating `.venv` and installing dependencies, run:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal (usually `http://localhost:8501`). The application has three tabs:

- **Detection:** Enter an English SMS message, select Naive Bayes or Logistic Regression, and view the predicted class and class probabilities.
- **Performance:** View saved evaluation metrics and visualizations.
- **About:** Read a short description of the project.

**Note:** The project already includes trained models in `models/` and evaluation outputs in `outputs/`. **Retraining is not required** to run the app.

## 7. Reproducing the Experiment and Retraining

1. Open `SMS_Spam_Detection.ipynb` in VS Code (with Jupyter support) or Jupyter Notebook.
2. Select the Python kernel from `.venv`.
3. Make sure the working directory is the **project root** so that the `data/`, `models/`, and `outputs/` paths resolve correctly.
4. Choose **Run All** to execute data loading, training, evaluation, and sample predictions in order.

**Warning:** Running the Notebook again **overwrites** the saved models in `models/` and evaluation artifacts in `outputs/`. Back up existing results if you need to preserve them.

## 8. Training and Test Results

The following results were measured on **1,035 held-out test messages**, which were not used for training. **Spam** is the positive class for Precision, Recall, and F1-score.

| Model | Threshold | Accuracy | Precision | Recall | F1-score | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | 0.50 | 96.33% | 100.00% | 70.99% | 83.04% | 904 | 0 | 38 | 93 |
| Logistic Regression | 0.50 | 96.33% | 100.00% | 70.99% | 83.04% | 904 | 0 | 38 | 93 |
| Multinomial Naive Bayes | 0.30 | 97.10% | 99.03% | 77.86% | 87.18% | 903 | 1 | 29 | 102 |
| Logistic Regression | 0.30 | 97.68% | 94.96% | 86.26% | 90.40% | 898 | 6 | 18 | 113 |

On the validation set, 0.30 gave each model its highest F1-score among 0.30, 0.40, and 0.50. On the test set, lowering the threshold detected more Spam and flagged more Ham as Spam (FP). Logistic Regression at 0.30 had the highest test F1-score in the table.

The Performance tab shows the threshold comparison from `outputs/threshold_metrics.csv` and charts at the 0.50 baseline. Notebook section 7.7 analyzes three test-set errors.

## 9. Limitations and Important Notes

- The reported metrics come from **one train/test split** with `random_state=42`; they do not constitute validation across multiple independent datasets.
- The models were trained on **English SMS**. Performance on Vietnamese messages or new spam patterns has not been evaluated.
- Only load `.joblib` files from trusted sources, as loading untrusted serialized models can execute unsafe code.
- The metrics and charts shown in the app describe the saved test-set results; they **do not guarantee accuracy on every real-world message**.
