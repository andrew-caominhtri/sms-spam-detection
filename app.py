import pandas as pd
from pathlib import Path
import streamlit as st
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

def load_model(model_name="naive_bayes"):
    model_files = {
        "naive_bayes": "naive_bayes_pipeline.joblib",
        "logistic_regression": "logistic_regression_pipeline.joblib",
    }
    if model_name not in model_files:
        raise ValueError(f"Mo hinh khong hop le: {model_name}")
    return joblib.load(MODELS_DIR / model_files[model_name])

def predict_message(model, message):
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    return prediction, dict(zip(model.classes_, probabilities))

@st.cache_resource
def get_model(model_name):
    """
    Tai mo hinh da huan luyen va luu vao cache.

    Args:
        model_name: Ten mo hinh can su dung.

    Returns:
        Pipeline da huan luyen.
    """
    return load_model(model_name)

# Cau hinh trang web
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide"
)


# Tieu de ung dung
st.title("SMS Spam Detection")
st.write("A Machine Learning application for detecting spam SMS messages.")


# Tao ba tab chinh
tab_detection, tab_performance, tab_about = st.tabs([
    "Detection",
    "Performance",
    "About"
])


# Tab 1: Phat hien tin nhan Spam
with tab_detection:

    st.header("SMS Spam Detection")

    st.write(
        "Enter an English SMS message to check "
        "whether it is Ham or Spam."
    )

    # 1. Nhap noi dung tin nhan
    message = st.text_area(
        "Enter your SMS message:",
        placeholder="Type your SMS message here...",
        height=150
    )

    # 2. Lua chon mo hinh
    model_options = {
        "Multinomial Naive Bayes": "naive_bayes",
        "Logistic Regression": "logistic_regression"
    }

    selected_model = st.selectbox(
        "Select Machine Learning Model:",
        options=list(model_options.keys())
    )

    # 3. Nut bat dau du doan
    if st.button("Detect Spam", type="primary"):

        # Kiem tra tin nhan rong
        if not message.strip():
            st.warning("Please enter an SMS message.")

        else:
            # Lay ten mo hinh tu lua chon
            model_name = model_options[selected_model]

            # Tai mo hinh da huan luyen
            model = get_model(model_name)

            # Du doan tin nhan
            prediction, probabilities = predict_message(
                model,
                message
            )

            # 4. Hien thi ket qua
            st.subheader("Prediction Result")

            if prediction == "spam":
                st.error("Spam Detected!")
            else:
                st.success("This message is Ham (Not Spam).")

            # 5. Hien thi xac suat
            st.write("Prediction Probabilities:")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Ham Probability",
                    f"{probabilities['ham'] * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Spam Probability",
                    f"{probabilities['spam'] * 100:.2f}%"
                )




# Tab 2: Hien thi ket qua danh gia mo hinh
with tab_performance:

    st.header("Model Performance")

    st.write(
        "Performance comparison between "
        "Multinomial Naive Bayes and Logistic Regression."
    )

    # Xac dinh duong dan file metrics.csv
    base_dir = Path(__file__).resolve().parent
    metrics_path = base_dir / "outputs" / "metrics.csv"

    # Kiem tra file co ton tai hay khong
    if metrics_path.exists():

        # Doc ket qua danh gia
        metrics_df = pd.read_csv(metrics_path)

        # Tao ban sao de hien thi du lieu dang phan tram
        display_df = metrics_df.copy()

        metric_columns = [
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        ]

        display_df[metric_columns] = (
            display_df[metric_columns] * 100
        ).round(2)

        # Hien thi bang so sanh
        st.subheader("Evaluation Metrics")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "All metrics are displayed as percentages. "
            "Precision, Recall and F1-score refer to the Spam class."
        )
        
        # Hien thi bieu do so sanh mo hinh
        st.subheader("Model Comparison")
        
        comparison_path = (
            base_dir / "outputs" / "model_comparison.png"
        )

        if comparison_path.exists():
            st.image(
                str(comparison_path),
                caption="Model Performance Comparison",
                use_column_width=True
            )
        else:
            st.warning("Model comparison chart not found.")
            
        # Hien thi Confusion Matrix
        st.subheader("Confusion Matrices")

        col1, col2 = st.columns(2)

        nb_path = (
            base_dir / "outputs" /
            "confusion_matrix_naive_bayes.png"
        )

        lr_path = (
            base_dir / "outputs" /
            "confusion_matrix_logistic_regression.png"
        )

        with col1:
            if nb_path.exists():
                st.image(
                    str(nb_path),
                    caption="Multinomial Naive Bayes",
                    use_column_width=True
                )
            else:
                st.warning("Naive Bayes chart not found.")

        with col2:
            if lr_path.exists():
                st.image(
                    str(lr_path),
                    caption="Logistic Regression",
                    use_column_width=True
                )
            else:
                st.warning("Logistic Regression chart not found.")

    else:
        st.warning("Evaluation results are not available.")



# Tab 3: Thong tin gioi thieu du an
with tab_about:
    st.header("About This Project")

    st.write(
        "SMS Spam Detection is a Machine Learning "
        "application designed to classify English SMS "
        "messages as Ham (legitimate) or Spam (unwanted)."
    )

    st.subheader("Dataset")
    st.write(
        "This project uses the UCI SMS Spam Collection "
        "dataset. After removing duplicate messages; "
        "5,169 SMS messages remain for model development."
    )

    st.subheader("Machine Learning Models")
    st.write(
        "Two classification algorithms are implemented "
        "and compared:"
    )
    st.markdown(
        "- Multinomial Naive Bayes\n"
        "- Logistic Regression"
    )

    st.subheader("Methodology")
    st.write(
        "SMS messages are transformed into numerical "
        "features using TF-IDF with unigram and bigram "
        "features. The dataset is divided into 80% "
        "training data and 20% testing data."
    )

    st.subheader("Evaluation Metrics")
    st.write(
        "The models are evaluated using Accuracy, "
        "Precision, Recall, F1-score and Confusion Matrix."
    )

    st.subheader("Technologies")
    st.markdown(
        "- Python\n"
        "- Pandas\n"
        "- Scikit-learn\n"
        "- Streamlit\n"
        "- Matplotlib and Seaborn\n"
        "- Joblib"
    )

    st.subheader("Project Repository")
    st.markdown(
        "[My GitHub]"
        "(https://github.com/andrew-caominhtri/sms-spam-detection)"
    )

