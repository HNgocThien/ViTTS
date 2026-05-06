# 2.4 Quy trình suy diễn (Inference)

Quá trình suy diễn (Inference) của F5-TTS là bước chuyển đổi văn bản đầu vào và âm thanh tham chiếu (prompt audio) thành tín hiệu giọng nói (waveform). Quá trình này được thực hiện qua hai giai đoạn chính: giải phương trình vi phân (ODE Solver) thông qua cơ chế Flow Matching để sinh ra các đặc trưng âm thanh, và sử dụng Vocoder để tổng hợp sóng âm thanh cuối cùng.

```mermaid
flowchart LR
    A[Text Input] --> B(Text Encoder)
    C[Audio Prompt] --> D(Audio Encoder)
    B --> E{DiT + Flow Matching\n(ODE Solver)}
    D --> E
    F[Noise] --> E
    E --> G[Mel-spectrogram]
    G --> H(Vocoder - Vocos)
    H --> I[Waveform Output]
    
    style E fill:#e1f5fe,stroke:#0288d1
    style H fill:#e8f5e9,stroke:#388e3c
```

## 2.4.1 Sampling trong Flow Matching

Khác với quá trình huấn luyện sử dụng hàm mất mát để tối ưu véc-tơ luồng (vector field), quá trình suy diễn trong F5-TTS thực chất là việc giải một phương trình vi phân thường (ODE) từ nhiễu thuần túy thành luồng dữ liệu mang ý nghĩa. 

Các bước cụ thể trong quá trình sampling:
1. **Khởi tạo trạng thái ban đầu**: Bắt đầu bằng một điểm lấy mẫu $x_0 \sim \mathcal{N}(0, I)$ biểu diễn cho nhiễu trắng (Gaussian noise).
2. **Điều kiện hóa (Conditioning)**: Đầu vào văn bản $y$ được mã hóa qua ConvNeXt V2 và âm thanh mẫu được đưa vào Diffusion Transformer (DiT). Thông tin này làm điều kiện hướng dẫn mô hình đoán nhận hướng đi (vector field) của dữ liệu.
3. **Giải phương trình ODE**: Sự tiến hóa của dữ liệu qua thời gian $t$ từ $t=0$ (nhiễu) đến $t=1$ (dữ liệu mục tiêu) được định nghĩa bởi phương trình:
   $$ \frac{dx_t}{dt} = v_\theta(x_t, t, y) $$
   Trong đó, $v_\theta$ là vận tốc do mạng DiT dự đoán. Tại mỗi bước (timestep), F5-TTS dùng các bộ giải ODE (như Euler Method hoặc Midpoint Method) để cập nhật giá trị của $x_t$.
4. **Classifier-Free Guidance (CFG)**: Để tăng cường độ chính xác và khả năng bám sát vào ngữ cảnh (đặc biệt là để duy trì sự đồng nhất của giọng người nói), F5-TTS có thể áp dụng CFG bằng cách nội suy tuyến tính giữa đầu ra dự đoán có điều kiện và không điều kiện.

Một trong những ưu điểm lớn nhất của phương pháp Flow Matching trong F5-TTS là cho phép sinh giọng nói chỉ với số lượng bước sampling (NFE - Number of Function Evaluations) rất nhỏ so với các mô hình Diffusion truyền thống. Việc giảm NFE mà vẫn giữ nguyên chất lượng giúp quá trình tổng hợp âm thanh đạt tốc độ suy diễn cao, đáp ứng được các ứng dụng thời gian thực.

## 2.4.2 Sinh tín hiệu âm thanh (Vocoding)

Đầu ra của mô hình sau khi sampling là một chuỗi các đặc trưng âm thanh dạng Mel-spectrogram. Tuy nhiên, tai người và các thiết bị phần cứng chỉ có thể phát được dữ liệu ở dạng tín hiệu sóng âm (waveform) trong miền thời gian. Do đó, bước cuối cùng là tái tạo tín hiệu âm thanh 1D từ Mel-spectrogram 2D.

F5-TTS lựa chọn **Vocos** [2] làm module Vocoder mặc định cho quá trình này. 
- **Động lực**: Khác với một số Vocoder truyền thống như HiFi-GAN sinh trực tiếp ra dạng sóng (waveform), Vocos hoạt động bằng cách dự đoán phổ tín hiệu (magnitude và phase) thông qua Inverse Short-Time Fourier Transform (ISTFT). Cơ chế này giúp giảm bớt lượng tính toán đáng kể trong khi vẫn tạo ra được chất lượng âm thanh vượt trội.
- **Vai trò**: Vocos giúp tái tạo hiệu quả các dải tần số cao, giải quyết tình trạng méo tiếng hoặc nhiễu, từ đó mang lại chất lượng giọng nói tự nhiên, mượt mà (đặc biệt là yếu tố prosody - nhịp điệu và ngữ điệu), hoàn toàn phù hợp với tốc độ sinh nhanh của DiT trong F5-TTS.

---

## Tài liệu tham khảo

- [1] Y. Chen et al., "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching," *arXiv preprint arXiv:2410.06885*, 2024.
- [2] J. Siuzdak, "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis," *arXiv preprint arXiv:2306.00814*, 2023.

---

## Phụ lục

- **ODE Solver (Bộ giải phương trình vi phân thường)**: Thuật toán toán học dùng để tìm xấp xỉ nghiệm của một phương trình vi phân qua nhiều bước lặp nhỏ. Trong hệ thống F5-TTS, nó dần dần dịch chuyển các điểm nhiễu trắng thành cấu trúc âm thanh có nghĩa.
- **Classifier-Free Guidance (CFG)**: Kỹ thuật giúp mô hình tạo sinh tập trung mạnh hơn vào các thông tin điều kiện đã cho (văn bản và prompt). Thuật toán này điều chỉnh hướng đi bằng cách đẩy lùi kết quả khỏi các dự đoán "không có ngữ cảnh" để làm sắc nét dự đoán cuối cùng.
- **Vocoder**: Mô hình chuyển đổi các dạng biểu diễn trung gian như Mel-spectrogram thành tín hiệu sóng âm thực tế để con người và thiết bị có thể nghe được.
- **Mel-spectrogram**: Biểu diễn trực quan về phổ tần số của âm thanh thay đổi theo thời gian, được tính toán dựa trên thang đo Mel để mô phỏng cách tai người cảm nhận sự chênh lệch cao độ.
