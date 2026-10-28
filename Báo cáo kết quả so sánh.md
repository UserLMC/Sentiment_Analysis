# BÁO CÁO THỰC HÀNH: PHÂN TÍCH CẢM XÚC TRÊN BỘ DỮ LIỆU IMDB (SENTIMENT ANALYSIS)

**Môn học:** Xử lý ngôn ngữ tự nhiên (NLP)  
**Chủ đề:** Xây dựng mô hình RNN, LSTM cho phân tích cảm xúc và so sánh với các phương pháp truyền thống (SVM, Logistic Regression)  
**Tập dữ liệu:** IMDB Movie Reviews (50,000 bài đánh giá: 25,000 Positive, 25,000 Negative)

---

## 1. Đặt vấn đề và Mục tiêu bài tập

Phân tích cảm xúc (Sentiment Analysis) là một trong những tác vụ nền tảng trong Xử lý ngôn ngữ tự nhiên (NLP). Bài tập yêu cầu:
1. Huấn luyện mô hình mạng nơ-ron hồi quy **Simple RNN** và **LSTM** (Long Short-Term Memory) để phân loại nhị phân đánh giá phim (tích cực hoặc tiêu cực).
2. Xây dựng và đánh giá hai mô hình học máy truyền thống làm cơ sở so sánh (**Baseline**): **Logistic Regression** và **Support Vector Machine (Linear SVM)** trên đặc trưng túi từ TF-IDF.
3. Phân tích, đối chiếu kết quả thực nghiệm giữa hai nhóm phương pháp dựa trên lý thuyết kiến trúc mạng chuỗi và suy giảm đạo hàm (Vanishing Gradient).

---

## 2. Thiết lập thực nghiệm & Tiền xử lý dữ liệu

### 2.1. Tập dữ liệu (IMDB Corpus)
* **Quy mô:** Gồm $50,000$ bài đánh giá phim tiếng Anh từ IMDB, nhãn được gán nhị phân ($0$: Tiêu cực, $1$: Tích cực).
* **Phân chia:** 
  * Tập huấn luyện (Train set): $25,000$ mẫu (12,500 Positive, 12,500 Negative).
  * Tập kiểm tra (Test set): $25,000$ mẫu (12,500 Positive, 12,500 Negative).
* **Kích thước từ điển (Vocabulary Size):** Giữ lại $10,000$ từ vựng xuất hiện phổ biến nhất để loại bỏ nhiễu và từ hiếm.

### 2.2. Xử lý đặc trưng cho từng nhóm mô hình
* **Nhóm học máy truyền thống (Logistic Regression, Linear SVM):**
  * Văn bản được chuyển đổi thành ma trận số học thông qua kỹ thuật **TF-IDF (Term Frequency - Inverse Document Frequency)** với số lượng đặc trưng tối đa là $10,000$.
* **Nhóm mạng nơ-ron hồi quy (Simple RNN, LSTM):**
  * Dữ liệu đầu vào được đệm (Padding) và cắt (Truncating) về độ dài cố định $MAX\_LEN = 200$ từ.
  * Tầng **Embedding**: Chuyển đổi mỗi từ thành một vector liên tục với số chiều $d = 128$.
  * Tầng hồi quy: $64$ đơn vị ẩn (hidden units) kèm cơ chế dropout $0.2$ để giảm quá khớp (overfitting).
  * Tầng đầu ra: $1$ nơ-ron với hàm kích hoạt Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, tối ưu hàm mất mát **Binary Cross-Entropy**.

---

## 3. Kết quả thực nghiệm

Kết quả kiểm tra trên tập test ($25,000$ mẫu chưa từng xuất hiện trong quá trình huấn luyện) thu được như sau:

| STT | Mô hình (Model) | Đặc trưng / Kiến trúc | Accuracy | F1-Score | Thời gian huấn luyện (ước tính) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | **Logistic Regression** | TF-IDF ($10,000$ features) | **0.8862** | **0.8868** | 0.61 giây |
| 2 | **SVM (LinearSVC)** | TF-IDF ($10,000$ features) | **0.8760** | **0.8754** | 1.21 giây |
| 3 | **Simple RNN** | Embedding ($128$) + SimpleRNN ($64$) | **0.8442** | **0.8519** | 56.33 giây |
| 4 | **LSTM** | Embedding ($128$) + LSTM ($64$) | **0.8656** | **0.8699** | 167.25 giây |

> **Nhận xét tổng quát:** Mô hình có hiệu năng cao nhất trên tập kiểm tra là **Logistic Regression** (Accuracy: $88.62\%$, F1-Score: $88.68\%$). Trong nhóm Deep Learning, **LSTM** vượt trội đáng kể so với **Simple RNN**. Độ chênh lệch giữa Accuracy và F1-Score trên cả 4 mô hình đều rất nhỏ ($< 0.008$), chứng minh mô hình phân lớp rất cân bằng và ổn định.

---

## 4. Phân tích và So sánh chuyên sâu

### 4.1. So sánh Simple RNN và LSTM
* **Về mặt định lượng:** LSTM vượt qua Simple RNN với mức chênh lệch tăng **+2.14%** về Accuracy ($0.8656$ so với $0.8442$) và **+1.80%** về F1-Score ($0.8699$ so với $0.8519$).
* **Giải thích nguyên nhân lý thuyết:**
  * **Hiện tượng suy giảm đạo hàm (Vanishing Gradient):** Trong Simple RNN, trạng thái ẩn được cập nhật theo công thức:
    $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
    Khi thực hiện lan truyền ngược qua thời gian (BPTT), đạo hàm gradient của hàm mất mát đối với các trọng số ở các bước thời gian đầu phụ thuộc vào tích của nhiều ma trận đạo hàm trung gian:
    $$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}$$
    Vì đạo hàm của $\tanh$ nằm trong khoảng $(0, 1]$ và các giá trị riêng của ma trận trọng số nhỏ hơn 1, lượng gradient này suy giảm theo hàm mũ về 0. Kết quả là mô hình "quên" hoàn toàn các từ và ngữ cảnh xuất hiện ở phần đầu của các bài đánh giá dài.
  * **Cơ chế khắc phục của LSTM:** LSTM giới thiệu trạng thái tế bào độc lập $C_t$ (Cell State) đóng vai trò như một "đường cao tốc truyền dẫn thông tin", kết hợp cùng hệ thống 3 cổng logic:
    * *Cổng quên (Forget gate):* $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
    * *Cổng vào (Input gate):* $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ cùng ứng viên nội dung mới $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$
    * *Cập nhật Cell State:* $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
    * *Cổng ra (Output gate):* $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ và $h_t = o_t \odot \tanh(C_t)$
    Nhờ phép cộng tuyến tính ($+$) trong quá trình cập nhật trạng thái tế bào, đạo hàm không bị triệt tiêu liên tục qua các phép nhân, cho phép LSTM ghi nhớ được ngữ cảnh cách xa hàng chục bước thời gian (chẳng hạn như mệnh đề tương phản đảo ngược ở cuối câu).

### 4.2. So sánh Deep Learning (RNN / LSTM) với Học máy truyền thống (Logistic Regression / SVM)
Mặc dù LSTM và RNN có kiến trúc phức tạp và xem xét thứ tự từ theo chuỗi, kết quả thực nghiệm cho thấy **Logistic Regression ($88.62\%$)** và **SVM ($87.60\%$)** lại đạt điểm số cao hơn. Hiện tượng này hoàn toàn hợp lý trong ngữ cảnh bài toán cụ thể bởi các nguyên nhân sau:

1. **Bản chất đặc thù của bài toán phân tích cảm xúc IMDB:**
   * Cảm xúc của bài đánh giá phim chủ yếu được chi phối bởi mật độ và cường độ của các từ biểu cảm mạnh (ví dụ: *terrible, brilliant, trash, masterpiece, awful, amazing*).
   * Phương pháp **TF-IDF** gán trọng số cực kỳ chính xác cho các từ khóa phân cực này trên toàn bộ văn bản. Trong không gian đặc trưng thưa nhưng nhiều chiều ($10,000$ chiều), Logistic Regression và Linear SVM tìm được siêu phẳng phân tách tối ưu gần như tức thì.
2. **Hạn chế cắt chuỗi (Sequence Truncation) của mạng hồi quy:**
   * Nhóm RNN và LSTM bị giới hạn ở độ dài $MAX\_LEN = 200$ từ nhằm tránh tràn bộ nhớ và quá tải tính toán. Nhiều bài đánh giá trong IMDB có độ dài vượt quá $300 - 500$ từ, do đó một phần thông tin (thường là phần kết luận cảm xúc ở cuối bài) đã bị cắt bớt (`truncation`). Trong khi đó, TF-IDF bao quát trọn vẹn $100\%$ từ vựng của toàn bài.
3. **Huấn luyện tầng Embedding từ đầu (From Scratch):**
   * Ma trận vector từ (Embedding layer) của RNN/LSTM trong bài thực hành này được khởi tạo ngẫu nhiên và chỉ học trên $25,000$ mẫu văn bản của tập train. Lượng dữ liệu này chưa đủ lớn để mô hình học được toàn diện biểu diễn ngữ nghĩa không gian sâu như các bộ từ điển tiền huấn luyện đồ sộ (Pre-trained Word Embeddings như GloVe hay Word2Vec).
4. **Hiệu suất tính toán:**
   * Logistic Regression và SVM huấn luyện xong chỉ trong vòng $1 - 3$ giây. Ngược lại, RNN và LSTM cần từ $40$ đến $70$ giây cho 5 epochs nhưng độ chính xác chưa vượt qua được baseline truyền thống.

---

## 5. Kết luận & Đề xuất cải tiến

### 5.1. Kết luận
* **Logistic Regression + TF-IDF** là một mô hình baseline cực kỳ mạnh mẽ, đơn giản và có chi phí tính toán thấp nhất nhưng mang lại hiệu năng cao nhất ($88.62\%$) cho bài toán Sentiment Analysis trên tập IMDB.
* **LSTM hoàn toàn áp đảo Simple RNN** ($86.56\%$ so với $84.42\%$), minh chứng rõ ràng cho vai trò quan trọng của các cơ chế cổng trong việc triệt tiêu hiện tượng suy giảm đạo hàm trên dữ liệu chuỗi.

### 5.2. Đề xuất cải tiến cho nhánh Deep Learning
Để mô hình học sâu có thể vượt mốc $90\%$ Accuracy và vượt qua các phương pháp truyền thống, các giải pháp kỹ thuật cần áp dụng tiếp theo bao gồm:
1. **Pre-trained Embeddings:** Tích hợp bộ vector nhúng sẵn như GloVe 100d/300d hoặc Word2Vec để mô hình nắm bắt ngữ nghĩa từ vựng phong phú ngay từ đầu.
2. **Bidirectional LSTM (BiLSTM):** Cho phép mạng đọc câu theo cả hai chiều (từ trái qua phải và từ phải qua trái) để hiểu ngữ cảnh toàn diện hơn.
3. **Mở rộng độ dài chuỗi:** Tăng $MAX\_LEN$ từ $200$ lên $400 - 500$ từ để hạn chế việc mất mát thông tin đối với các bài bình luận dài.
4. **Cơ chế Attention / Transformer:** Ứng dụng lớp Self-Attention hoặc kiến trúc BERT để mô hình có thể trực tiếp tập trung vào các từ mang tính cảm xúc ở bất kỳ vị trí nào trong câu mà không bị giới hạn bởi độ dài lan truyền chuỗi.