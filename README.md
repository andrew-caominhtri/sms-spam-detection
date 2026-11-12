# SMS Spam Detection – Phát hiện tin nhắn SMS rác bằng học máy

## 1. Giới thiệu đề tài

**Tên đề tài:** SMS Spam Detection – **Xây dựng ứng dụng phát hiện tin nhắn SMS rác bằng Machine Learning (Học máy)**.

Đồ án xây dựng hệ thống phân loại nội dung tin nhắn SMS thành hai nhóm:

- **Ham:** Tin nhắn hợp lệ, không phải tin nhắn rác.
- **Spam:** Tin nhắn rác, thường có nội dung quảng cáo, mời chào hoặc lừa đảo.

Hệ thống sử dụng **TF-IDF** để biểu diễn văn bản và so sánh hai thuật toán **Multinomial Naive Bayes** và **Logistic Regression**. Phần thực nghiệm được trình bày trong **Jupyter Notebook**; ứng dụng **Streamlit** cho phép nhập tin nhắn và xem kết quả dự đoán trực tiếp.

**Phạm vi:** Mô hình hiện được huấn luyện và đánh giá trên **tin nhắn SMS tiếng Anh**. Chưa có kết quả kiểm chứng cho tiếng Việt.

## 2. Bộ dữ liệu

- **Tên:** SMS Spam Collection.
- **Nguồn:** UCI Machine Learning Repository – [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
- **Dữ liệu sử dụng:** `data/sms+spam+collection/SMSSpamCollection` (hai cột: `label`, `message`, phân cách bằng tab).
- **Số mẫu ban đầu:** 5.572 SMS, gồm 4.825 Ham và 747 Spam.
- **Sau loại bỏ tin nhắn trùng lặp:** 5.169 SMS, gồm 4.516 Ham và 653 Spam.
- **Chia dữ liệu:** 80% Train (4.135 mẫu), 20% Test (1.034 mẫu), phân tầng theo nhãn (`stratify`), `random_state=42`.

Thư mục `data/VN-dataset/` là dữ liệu tiếng Việt được lưu trong dự án để tham khảo, **không được sử dụng để huấn luyện hoặc đánh giá hai mô hình hiện tại**.

## 3. Quy trình thực hiện

1. Đọc dữ liệu SMS và loại bỏ bản ghi trùng lặp.
2. Chia tập Train/Test theo tỷ lệ 80/20, bảo toàn tương đối tỷ lệ nhãn.
3. Trích xuất đặc trưng **TF-IDF** với unigram và bigram (`ngram_range=(1, 2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`). Bộ TF-IDF chỉ được `fit` trên tập Train.
4. Huấn luyện hai Pipeline: **TF-IDF + Multinomial Naive Bayes** và **TF-IDF + Logistic Regression**.
5. Đánh giá trên tập Test bằng Accuracy, Precision, Recall, F1-score và Confusion Matrix; **Spam là lớp dương**.
6. Lưu mô hình `.joblib`, kết quả `.csv` và biểu đồ `.png`; sử dụng các file này trong ứng dụng Streamlit.

## 4. Cấu trúc thư mục

```text
sms-spam-detection/
├── SMS_Spam_Detection.ipynb                            # Toàn bộ thực nghiệm: dữ liệu, TF-IDF, huấn luyện, đánh giá, lưu mô hình
├── app.py                                              # Giao diện Streamlit: Detection, Performance, About
├── requirements.txt                                    # Các thư viện Python và phiên bản sử dụng
├── README.md                                           # Giới thiệu, hướng dẫn cài đặt và kết quả đồ án
├── .gitignore                                          # Danh sách file/thư mục không đưa vào Git
├── data/
│   ├── sms+spam+collection/
│   │   ├── SMSSpamCollection                           # Dataset tiếng Anh dùng trong thực nghiệm
│   │   └── readme                                      # Thông tin kèm theo dataset gốc
│   └── VN-dataset/
│       ├── full_dataset.csv                            # Dữ liệu tiếng Việt lưu tham khảo, không sử dụng
│       ├── train.csv                                   # Dữ liệu tiếng Việt lưu tham khảo, không sử dụng
│       └── test.csv                                    # Dữ liệu tiếng Việt lưu tham khảo, không sử dụng
├── models/
│   ├── naive_bayes_pipeline.joblib                     # Pipeline TF-IDF + Multinomial Naive Bayes đã huấn luyện
│   └── logistic_regression_pipeline.joblib             # Pipeline TF-IDF + Logistic Regression đã huấn luyện
├── outputs/
│   ├── metrics.csv                                     # Bảng chỉ số đánh giá hai mô hình
│   ├── confusion_matrix_naive_bayes.png                # Ma trận nhầm lẫn Naive Bayes
│   ├── confusion_matrix_logistic_regression.png        # Ma trận nhầm lẫn Logistic Regression
│   └── model_comparison.png                            # Biểu đồ so sánh các chỉ số
└── report/                                             # Thư mục dành cho báo cáo đồ án
```

## 5. Cài đặt môi trường

**Yêu cầu:** Python 3.11, PowerShell (Windows) và mã nguồn dự án. Mở Terminal **tại thư mục gốc** `sms-spam-detection`.

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện của dự án
python -m pip install -r requirements.txt

# Kiểm tra phụ thuộc
python -m pip check
```

Để **mở và chạy Notebook**, cài thêm (không bắt buộc nếu chỉ dùng ứng dụng Streamlit):

```powershell
python -m pip install notebook ipykernel
```

Có thể mở Notebook bằng VS Code với tiện ích Jupyter, chọn Python interpreter thuộc `.venv`.

## 6. Chạy ứng dụng Streamlit

Tại thư mục gốc, sau khi kích hoạt `.venv` và cài thư viện:

```powershell
streamlit run app.py
```

Mở địa chỉ local được Streamlit hiển thị trong Terminal (thường là `http://localhost:8501`). Ứng dụng có ba tab:

- **Detection:** Nhập nội dung SMS tiếng Anh, chọn Naive Bayes hoặc Logistic Regression và xem nhãn dự đoán cùng xác suất.
- **Performance:** Xem bảng chỉ số đánh giá và các biểu đồ đã lưu.
- **About:** Xem thông tin giới thiệu dự án.

**Lưu ý:** Thư mục `models/` và `outputs/` đã có sẵn trong bản dự án, nên **không cần huấn luyện lại** để chạy ứng dụng.

## 7. Chạy lại thực nghiệm và huấn luyện

1. Mở `SMS_Spam_Detection.ipynb` trong VS Code (Jupyter) hoặc Jupyter Notebook.
2. Chọn kernel Python của `.venv`.
3. Đảm bảo đang làm việc tại **thư mục gốc dự án** để đường dẫn `data/`, `models/`, `outputs/` hoạt động đúng.
4. Chọn **Run All** để chạy lần lượt từ đọc dữ liệu đến huấn luyện, đánh giá và thử dự đoán.

**Cảnh báo:** Chạy lại Notebook sẽ **ghi đè** các mô hình trong `models/` và các kết quả trong `outputs/`. Nếu cần giữ kết quả cũ, hãy sao lưu trước.

## 8. Kết quả huấn luyện và đánh giá

Kết quả được đo trên **1.034 SMS thuộc tập Test**, không dùng để huấn luyện. Lớp dương trong các phép tính Precision, Recall và F1-score là **Spam**.

| Chỉ số | Multinomial Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | **96,32%** | 96,23% |
| Precision (Spam) | 100,00% | 100,00% |
| Recall (Spam) | **70,99%** | 70,23% |
| F1-score (Spam) | **83,04%** | 82,51% |

Ma trận nhầm lẫn theo thứ tự nhãn `[ham, spam]`:

| Mô hình | TN (Ham đúng) | FP (Ham bị báo Spam) | FN (Spam bị bỏ sót) | TP (Spam đúng) |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 903 | 0 | 38 | 93 |
| Logistic Regression | 903 | 0 | 39 | 92 |

**Nhận xét:** Hai mô hình có độ chính xác tổng thể gần tương đương; Naive Bayes nhỉnh hơn nhẹ về Recall và F1-score trên tập Test này. Cả hai không báo nhầm tin Ham thành Spam trong tập Test, nhưng vẫn **bỏ sót 38–39 tin Spam**. Vì vậy, không nên diễn giải Accuracy hoặc Precision cao thành khả năng phát hiện Spam hoàn hảo.

Kết quả chi tiết và biểu đồ được lưu tại `outputs/metrics.csv`, `outputs/model_comparison.png` và hai file `confusion_matrix_*.png`.

## 9. Giới hạn và lưu ý

- Kết quả trên là kết quả của **một lần chia Train/Test** với `random_state=42`, chưa phải đánh giá trên nhiều bộ dữ liệu độc lập.
- Dữ liệu huấn luyện là **SMS tiếng Anh**; chưa đánh giá khả năng nhận diện tin nhắn tiếng Việt hoặc các kiểu Spam mới.
- Không tải file `.joblib` từ nguồn không đáng tin cậy, vì việc nạp mô hình có thể thực thi mã không an toàn.
- Các chỉ số và biểu đồ trong giao diện là kết quả của tập Test đã lưu, **không phải độ chính xác bảo đảm cho mọi tin nhắn thực tế**.
