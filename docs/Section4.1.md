# 4. Thực Nghiệm và Đánh Giá

Chương này trình bày chi tiết về quá trình thực nghiệm, từ việc thiết lập môi trường, chuẩn bị dữ liệu cho đến các kết quả đánh giá chất lượng giọng nói được sinh ra từ hệ thống F5-TTS sau khi fine-tuning.

## 4.1 Thiết lập thực nghiệm

Để đánh giá hiệu năng và khả năng thích nghi (speaker adaptation) của mô hình F5-TTS đối với giọng nói tiếng Việt cá nhân, một môi trường thực nghiệm tiêu chuẩn đã được thiết lập. Quá trình này bao gồm việc xây dựng một tập dữ liệu chuyên biệt và cấu hình hệ thống phần cứng, phần mềm tối ưu cho việc huấn luyện Diffusion Transformer (DiT).

### 4.1.1 Tập dữ liệu sử dụng

Thay vì sử dụng các tập dữ liệu công khai có sẵn, nghiên cứu này tự xây dựng tập dữ liệu giọng nói cá nhân mang tên `thien_dataset`. Việc tự thu thập giúp kiểm soát hoàn toàn chất lượng âm thanh đầu vào và tối ưu hóa cho bài toán Voice Cloning của chính tác giả.

**Quy mô và Đặc tả dữ liệu:**
- **Số lượng mẫu:** 696 cặp âm thanh - văn bản (audio-text pairs).
- **Định dạng âm thanh:** Tín hiệu âm thanh được thu trong môi trường kín, ít tạp âm, lưu trữ dưới định dạng WAV với tần số lấy mẫu (Sampling Rate) 24kHz, 16-bit PCM, kênh đơn (mono). Định dạng này hoàn toàn tương thích với yêu cầu đầu vào của bộ trích xuất đặc trưng âm thanh ConvNeXt V2 trong hệ thống F5-TTS.
- **Siêu dữ liệu (Metadata):** Các cặp dữ liệu được ánh xạ thông qua tệp `thien_dataset-metadata.txt` với định dạng `[tên_file_âm_thanh]|[nội_dung_văn_bản]`.

**Tính đa dạng của ngữ liệu:**
Để đảm bảo mô hình F5-TTS học được các phân phối xác suất phức tạp của ngữ điệu tiếng Việt (prosody) thông qua cơ chế Flow Matching, tập lệnh đọc (prompts) được thiết kế bao phủ nhiều kịch bản ngữ âm:
1. **Đa dạng về loại câu:** Bao gồm câu trần thuật, câu hỏi, và câu cảm thán nhằm giúp mô hình học cách thay đổi cao độ (pitch) ở cuối câu một cách tự nhiên.
2. **Đa dạng về độ dài:** Từ các câu mệnh lệnh ngắn (dưới 10 từ) đến các câu ghép, đoạn văn kể chuyện, mô tả dài (hơn 40 từ), kiểm tra khả năng duy trì nhịp điệu và tránh lỗi sinh chữ ảo (hallucination) của DiT khi xử lý chuỗi dài.
3. **Thành phần phức tạp:** Chứa các con số (VD: *2026*, *192.168.1.1*), ngày tháng, từ ngoại lai (VD: *tts*, *ai*), và ký tự đặc biệt (*@, #, %*) để đánh giá module chuẩn hóa văn bản (Text Normalization) của hệ thống.

### 4.1.2 Môi trường phần cứng và phần mềm

Kiến trúc F5-TTS, với backbone là mạng Diffusion Transformer (DiT), đòi hỏi khối lượng tính toán lớn, đặc biệt là dung lượng bộ nhớ đồ họa (VRAM) trong quá trình fine-tuning do phải lưu trữ các bản đồ đặc trưng (feature maps) và lan truyền ngược (backpropagation) qua nhiều bước thời gian (timesteps).

**Cấu hình phần cứng:**
- **GPU:** Thực nghiệm được tiến hành trên máy chủ trang bị NVIDIA RTX 3090 (Hoặc 4090) với 24GB VRAM. Đây là mức cấu hình được khuyến nghị để có thể huấn luyện toàn bộ tham số (Full Fine-tuning) của mạng DiT mà không bị lỗi tràn bộ nhớ (Out-of-Memory).
- **CPU & RAM:** Vi xử lý đa nhân, bộ nhớ RAM hệ thống từ 32GB trở lên để đảm bảo không nghẽn cổ chai (bottleneck) trong quá trình tiền xử lý và nạp dữ liệu (data loading).

**Cấu hình phần mềm:**
- **Hệ điều hành:** Ubuntu 22.04 LTS (hoặc Windows 11 tích hợp WSL2).
- **Môi trường lập trình:** Python 3.10+, PyTorch 2.x (hỗ trợ tính toán tensor tăng tốc trên GPU).
- **Trình điều khiển (Drivers):** CUDA Toolkit 11.8/12.x và thư viện cuDNN tương ứng.
- **Kỹ thuật tối ưu:** Để huấn luyện hiệu quả trên GPU 24GB VRAM, hệ thống áp dụng tích hợp các kỹ thuật: bộ tối ưu hóa **8-bit AdamW**, **Gradient Accumulation** (tích lũy gradient qua nhiều bước để mô phỏng batch size lớn), và **Gradient Checkpointing** (giảm thiểu lưu trữ activation trung gian).

---

## Phụ Lục (Appendix)

> **Giải thích các thuật ngữ chuyên ngành xuất hiện trong phần này:**

- **Tần số lấy mẫu (Sampling Rate):** Số lần lấy mẫu tín hiệu âm thanh trong một giây. 24kHz nghĩa là có 24,000 mẫu được lấy mỗi giây, đủ để tái tạo rõ nét các tần số giọng nói con người (thường nằm dưới ngưỡng 12kHz theo định lý Nyquist).
- **16-bit PCM (Pulse-Code Modulation):** Phương pháp biểu diễn tín hiệu âm thanh analog dưới dạng digital không nén. Độ sâu bit (bit depth) là 16-bit, cung cấp dải động (dynamic range) khoảng 96dB, cho chất lượng âm thanh tiêu chuẩn phòng thu.
- **VRAM (Video Random Access Memory):** Bộ nhớ đồ họa của thiết bị GPU. Trong Deep Learning, VRAM dùng để chứa các tham số (parameters) của mô hình, dữ liệu đầu vào (batch data), và các giá trị kích hoạt (activations) sinh ra trong quá trình huấn luyện.
- **Out-of-Memory (OOM):** Lỗi xảy ra khi dữ liệu cần xử lý vượt quá giới hạn dung lượng VRAM vật lý của GPU, khiến chương trình bị buộc dừng.
- **Gradient Checkpointing:** Một kỹ thuật đánh đổi giữa thời gian tính toán và không gian lưu trữ. Thay vì lưu trữ toàn bộ các giá trị trung gian ở quá trình truyền xuôi (forward pass) để phục vụ cho việc tính đạo hàm ở quá trình truyền ngược (backward pass), kỹ thuật này chỉ lưu một số điểm "chốt" (checkpoints) và sẽ tính toán lại các phần còn thiếu khi cần, giúp giảm đáng kể lượng VRAM tiêu thụ.
- **8-bit AdamW:** Phiên bản tối ưu hóa của thuật toán AdamW, trong đó các trạng thái của optimizer được lượng tử hóa (quantized) xuống 8-bit thay vì 32-bit như thông thường, giúp tiết kiệm đáng kể dung lượng bộ nhớ hệ thống trong quá trình fine-tuning mô hình lớn.
