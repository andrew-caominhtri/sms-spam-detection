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
- **Original dataset:** 5,572 messages: 4,825 Ham and 747 Spam.
- **After removing duplicate messages:** 5,169 messages: 4,516 Ham and 653 Spam.
- **Data split:** 80% training (4,135 messages) and 20% testing (1,034 messages), stratified by class label, with `random_state=42`.

The `data/VN-dataset/` directory contains Vietnamese-language data retained for reference. **It is not used to train or evaluate the current models.**

## 3. Machine Learning Workflow

1. Load the SMS dataset and remove duplicate records.
2. Split the cleaned data into training and test sets (80/20), preserving the approximate class distribution.
3. Extract **TF-IDF** features with unigrams and bigrams (`ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`). The vectorizer is fitted **only on the training set**.
4. Train two pipelines: **TF-IDF + Multinomial Naive Bayes** and **TF-IDF + Logistic Regression**.
5. Evaluate both models on the held-out test set using Accuracy, Precision, Recall, F1-score, and confusion matrices. **Spam is treated as the positive class.**
6. Save the trained pipelines as `.joblib` files, evaluation metrics as `.csv`, and charts as `.png`. The Streamlit app loads these artifacts.

## 4. Project Structure

```text
sms-spam-detection/
├── SMS_Spam_Detection.ipynb         # Full ML experiment: data, TF-IDF, training, evaluation, model export
├── app.py                           # Streamlit interface: Detection, Performance, About
├── requirements.txt                 # Python dependencies and pinned versions
├── README.md                        # Project overview, setup instructions, and results
├── .gitignore                       # Files and folders excluded from Git tracking
├── data/
│   ├── sms+spam+collection/
│   │   ├── SMSSpamCollection        # English SMS dataset used in the experiment
│   │   └── readme                   # Original dataset documentation
│   └── VN-dataset/
│       ├── full_dataset.csv         # Reference Vietnamese dataset; not used
│       ├── train.csv                # Reference Vietnamese dataset; not used
│       └── test.csv                 # Reference Vietnamese dataset; not used
├── models/
│   ├── naive_bayes_pipeline.joblib  # Trained TF-IDF + Multinomial Naive Bayes pipeline
│   └── logistic_regression_pipeline.joblib  # Trained TF-IDF + Logistic Regression pipeline
├── outputs/
│   ├── metrics.csv                  # Evaluation metrics for both models
│   ├── confusion_matrix_naive_bayes.png        # Naive Bayes confusion matrix
│   ├── confusion_matrix_logistic_regression.png # Logistic Regression confusion matrix
│   └── model_comparison.png         # Model performance comparison chart
└── report/                          # Directory reserved for the project report
```

## 5. Environment Setup

**Requirements:** Python 3.11, Windows PowerShell, and the project source files. Open a terminal in the **project root** (`sms-spam-detection`).

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
streamlit run app.py
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

The following results were measured on **1,034 held-out test messages**, which were not used for training. **Spam** is the positive class for Precision, Recall, and F1-score.

| Metric | Multinomial Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | **96.32%** | 96.23% |
| Precision (Spam) | 100.00% | 100.00% |
| Recall (Spam) | **70.99%** | 70.23% |
| F1-score (Spam) | **83.04%** | 82.51% |

Confusion matrix counts, using class order `[ham, spam]`:

| Model | TN (Correct Ham) | FP (Ham Flagged as Spam) | FN (Missed Spam) | TP (Correct Spam) |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 903 | 0 | 38 | 93 |
| Logistic Regression | 903 | 0 | 39 | 92 |

**Interpretation:** Both models achieve similar overall accuracy. Naive Bayes performs slightly better on Recall and F1-score on this test split. Neither model incorrectly flagged a Ham message as Spam in this test set, but they still **missed 38–39 Spam messages**. High Accuracy or Precision therefore should not be interpreted as perfect spam detection.

Detailed metrics and figures are available in `outputs/metrics.csv`, `outputs/model_comparison.png`, and the two `confusion_matrix_*.png` files.

## 9. Limitations and Important Notes

- The reported metrics come from **one train/test split** with `random_state=42`; they do not constitute validation across multiple independent datasets.
- The models were trained on **English SMS**. Performance on Vietnamese messages or new spam patterns has not been evaluated.
- Only load `.joblib` files from trusted sources, as loading untrusted serialized models can execute unsafe code.
- The metrics and charts shown in the app describe the saved test-set results; they **do not guarantee accuracy on every real-world message**.
