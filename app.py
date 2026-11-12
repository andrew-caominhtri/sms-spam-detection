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


def explain_prediction(model, message, top_n=5):
    """
    Tinh muc do dong gop cua cac dac trung TF-IDF
    vao du doan Spam so voi Ham.
    """

    # Lay TF-IDF va classifier tu Pipeline
    vectorizer = model.steps[0][1]
    classifier = model.steps[-1][1]

    # Bien doi tin nhan thanh vector TF-IDF
    features = vectorizer.transform([message])

    # Lay ten cac tu/cum tu
    feature_names = vectorizer.get_feature_names_out()

    # Xac dinh vi tri nhan Spam va Ham
    classes = list(classifier.classes_)
    spam_idx = classes.index("spam")
    ham_idx = classes.index("ham")

    # Tinh trong so cua tung dac trung
    if hasattr(classifier, "feature_log_prob_"):
        # Multinomial Naive Bayes
        weights = (
            classifier.feature_log_prob_[spam_idx]
            - classifier.feature_log_prob_[ham_idx]
        )

    elif hasattr(classifier, "coef_"):
        # Logistic Regression nhi phan
        spam_sign = 1 if classes[1] == "spam" else -1
        weights = spam_sign * classifier.coef_[0]

    else:
        raise ValueError("Unsupported model for explanation.")

    # Chi lay cac dac trung xuat hien trong tin nhan
    contributions = []

    for index, value in zip(features.indices, features.data):
        contribution = float(value * weights[index])

        contributions.append({
            "Feature": feature_names[index],
            "Contribution": contribution
        })

    # Sap xep theo muc do anh huong tuyet doi
    contributions.sort(
        key=lambda item: abs(item["Contribution"]),
        reverse=True
    )

    return contributions[:top_n]


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

    # Danh sach SMS mau de demo
    sample_messages = {
        "Custom Message": "",
        "Normal SMS": "Hi, are we still meeting at the library at 3 pm tomorrow?",
        "Spam SMS": "Congratulations! You have won a free prize. Claim now!",
        "Challenging SMS": "Hi, you have been selected for a special offer. Contact us today!"
    }

    # Lua chon SMS mau
    selected_sample = st.selectbox(
        "Choose a sample SMS:",
        options=list(sample_messages.keys())
    )

    # Cap nhat noi dung khi thay doi SMS mau
    if "sms_message" not in st.session_state:
        st.session_state.sms_message = ""

    if "previous_sample" not in st.session_state:
        st.session_state.previous_sample = "Custom Message"

    if selected_sample != st.session_state.previous_sample:
        st.session_state.sms_message = sample_messages[selected_sample]
        st.session_state.previous_sample = selected_sample

    # O nhap tin nhan
    message = st.text_area(
        "Enter your SMS message:",
        placeholder="Type your SMS message here...",
        height=150,
        key="sms_message"
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
    
    # 2.1. Dieu chinh nguong phan loai Spam
    spam_threshold = st.slider(
        "Spam Classification Threshold (Ngưỡng phân loại Spam):",
        min_value=0.3,
        max_value=0.5,
        value=0.5,
        step=0.1,
        help=(
            "Lower thresholds may detect more Spam but can "
            "increase false positives. / "
            "Ngưỡng thấp có thể phát hiện thêm Spam nhưng "
            "cũng có thể làm tăng số tin Ham bị nhận nhầm."
        )
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
                
            
            # 6. Giai thich ket qua du doan
            st.subheader(
                "Prediction Explanation (Giải thích kết quả dự đoán)"
            )

            st.write(
                "Top TF-IDF features influencing the prediction "
                "toward Spam or Ham."
            )

            st.caption(
                "Các từ hoặc cụm từ có mức đóng góp lớn nhất "
                "vào kết quả phân loại tin nhắn."
            )

            explanations = explain_prediction(
                model,
                message,
                top_n=5
            )

            if explanations:
                explanation_df = pd.DataFrame(explanations)

                explanation_df["Direction"] = (
                    explanation_df["Contribution"].apply(
                        lambda x: "Spam" if x > 0
                        else "Ham" if x < 0
                        else "Neutral"
                    )
                )

                explanation_df = explanation_df.rename(
                    columns={
                        "Feature": "Feature (Từ/Cụm từ)",
                        "Contribution": "Contribution (Đóng góp)",
                        "Direction": "Direction (Xu hướng)"
                    }
                )

                st.dataframe(
                    explanation_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("**Result Interpretation / Giải thích kết quả:**")

                st.markdown(
                    "- **Positive (+) / Dương:** "
                    "Đặc trưng góp phần làm tăng xu hướng dự đoán Spam.\n"
                    "- **Negative (-) / Âm:** "
                    "Đặc trưng góp phần làm tăng xu hướng dự đoán Ham.\n"
                    "- **Contribution / Đóng góp:** "
                    "Giá trị đóng góp vào điểm log-odds, "
                    "không phải phần trăm xác suất."
                )

                st.info(
                    "The final prediction also depends on other "
                    "features and the model's intercept or prior.\n\n"
                    "Dự đoán cuối cùng còn phụ thuộc vào các "
                    "đặc trưng khác và hệ số chặn hoặc xác suất "
                    "tiên nghiệm của mô hình."
                )

            else:
                st.info(
                    "No matching TF-IDF features found.\n\n"
                    "Không tìm thấy đặc trưng TF-IDF phù hợp "
                    "trong tin nhắn."
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

