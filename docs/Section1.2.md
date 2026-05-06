# 1.2 Mô Hình Xác Suất Trong TTS

Chương trước đặt bài toán TTS là ánh xạ text → audio. Chương này hình thức hóa ánh xạ đó bằng ngôn ngữ xác suất — nền tảng toán học để hiểu tại sao các mô hình sinh như Diffusion Models và Flow Matching lại hoạt động hiệu quả.

## 1.2.1 Bài Toán TTS Dưới Góc Nhìn Xác Suất

Do tính **one-to-many** của TTS (cùng văn bản, nhiều cách phát âm hợp lệ), bài toán được phát biểu là học **phân phối xác suất có điều kiện** \[[1]\]:

$$p_\theta(\mathbf{x} \mid \mathbf{y}) \approx p_{\text{data}}(\mathbf{x} \mid \mathbf{y})$$

Trong đó:
- $\mathbf{y}$: chuỗi văn bản (ký tự, phoneme, hoặc byte-level token)
- $\mathbf{x}$: mel-spectrogram tương ứng thuộc không gian liên tục $\mathcal{X}$
- $\theta$: tham số mạng nơ-ron cần học

Quá trình **suy diễn** (inference) là lấy mẫu từ phân phối đã học:

$$\mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{y})$$

Điều này giải thích tại sao cùng một văn bản, mô hình có thể sinh ra các âm thanh khác nhau — mỗi lần sample cho một điểm khác nhau từ $p_\theta$.

## 1.2.2 Phân Phối Dữ Liệu Âm Thanh

Phân phối $p_{\text{data}}(\mathbf{x}|\mathbf{y})$ có ba đặc điểm khiến nó khó học trực tiếp \[[2, 3]\]:

1. **Số chiều cực lớn**: Một giây âm thanh biểu diễn dưới dạng mel-spectrogram 80 dải × 86 frames ≈ 6,880 chiều. Không thể mô hình hóa tường minh trong không gian này.

2. **Phân phối đa đỉnh (multimodal)**: Cùng một câu có thể nói vui, buồn, nhanh, chậm — mỗi phong cách tạo thành một cụm (mode) riêng trong $\mathcal{X}$. Phương pháp hồi quy chuẩn bị mất mát $L_2$ sẽ lấy trung bình các mode → âm thanh mờ nhạt (oversmoothed).

3. **Đa tạp giả thuyết (manifold hypothesis)** \[[4]\]: Dữ liệu âm thanh tự nhiên không phân bố rải rác toàn bộ $\mathcal{X}$, mà tập trung trên một đa tạp con chiều thấp hơn. Mô hình phải học cách đặt sample ra đúng trên đa tạp này — nếu lệch ra ngoài, âm thanh sẽ nghe bị nhiễu hoặc biến dạng.

## 1.2.3 Hàm Mục Tiêu — Học Phân Phối Nào Tốt Nhất?

Các thế hệ mô hình TTS khác nhau ở cách chọn *objective function* để khớp $p_\theta$ với $p_{\text{data}}$.

### Maximum Likelihood Estimation (MLE) — KL Divergence

Cách truyền thống nhất: tối thiểu hóa **Forward KL Divergence** \[[5]\]:

$$D_{\text{KL}}(p_{\text{data}} \| p_\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \left[\log \frac{p_{\text{data}}(x)}{p_\theta(x)}\right]$$

MLE khuyến khích $p_\theta$ "bao phủ" toàn bộ support của $p_{\text{data}}$ → hiện tượng *mean-seeking* → giọng nói sinh ra đều đều, thiếu sắc thái. Đây là hạn chế chính của Tacotron \[[6]\] và các mô hình seq2seq sớm.

### Score Matching — Nền Tảng Của Diffusion Models

Thay vì mô hình hóa $p(x)$ trực tiếp, **Score Matching** \[[7]\] mô hình hóa đạo hàm log-density (gọi là *score function*):

$$s_\theta(x) \approx \nabla_x \log p_{\text{data}}(x)$$

Đây là nền tảng của Diffusion Models: mạng nơ-ron học dự đoán hướng gradient của phân phối dữ liệu, không cần biết hàm mật độ tường minh. Chất lượng sinh ra cao hơn MLE nhưng cần nhiều bước suy diễn (T=1000).

### Optimal Transport — Nền Tảng Của Flow Matching

**Khoảng cách Wasserstein** \[[8]\] đo chi phí tối thiểu để "vận chuyển" xác suất từ phân phối này sang phân phối khác:

$$W_2(P, Q) = \left(\inf_{\pi \in \Gamma(P,Q)} \int \|x - y\|^2 \, d\pi(x, y)\right)^{1/2}$$

Lý thuyết Optimal Transport chỉ ra rằng với phân phối Gaussian nguồn $P = \mathcal{N}(0, \mathbf{I})$ và phân phối dữ liệu đích $Q = p_{\text{data}}$, tồn tại một **đường vận chuyển thẳng** (linear interpolation path) tối ưu \[[9]\]:

$$x_t = (1-t)\, x_0 + t\, x_1, \quad t \in [0,1]$$

**Flow Matching** \[[10]\] học vector field $v_\theta(x_t, t)$ để mô hình hóa đường đi này:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\, x_t,\, \mathbf{y}}\, \bigl\|v_\theta(x_t, t, \mathbf{y}) - u_t(x_t \mid x_1)\bigr\|^2$$

Ưu điểm so với Diffusion: gradient có ý nghĩa ngay cả khi $P$ và $Q$ không giao nhau (thường xảy ra đầu training), và suy diễn chỉ cần giải một ODE với 10–32 bước thay vì 1000 bước \[[10]\].

---

## Kết Luận Phần 1.2

| Phương pháp | Objective | Chất lượng | Tốc độ suy diễn |
|---|---|---|---|
| MLE (Tacotron) | Forward KL | Trung bình (oversmoothed) | Nhanh |
| Score Matching (Diffusion) | Fisher Divergence | Cao | Chậm (1000 bước) |
| Flow Matching (F5-TTS) | Conditional FM | Cao | Nhanh (10–32 bước) |

Flow Matching là lựa chọn của F5-TTS vì đạt được cân bằng tốt nhất giữa chất lượng và tốc độ suy diễn — hai tiêu chí then chốt cho ứng dụng TTS thực tế.

---

## Tài Liệu Tham Khảo

| # | Tác giả | Năm | Tiêu đề | Venue |
|---|---------|-----|---------|-------|
| [1] | Chen, Y., Yue, Z., et al. | 2024 | *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching* | arXiv:2410.06885 |
| [2] | Shen, J., Pang, R., et al. | 2018 | *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions* | ICASSP 2018 |
| [3] | Taylor, P. | 2009 | *Text-to-Speech Synthesis* | Cambridge University Press |
| [4] | Fefferman, C., Mitter, S., Narayanan, H. | 2016 | *Testing the Manifold Hypothesis* | J. Amer. Math. Soc. |
| [5] | Goodfellow, I., et al. | 2016 | *Deep Learning* | MIT Press |
| [6] | Wang, Y., Skerry-Ryan, R.J., et al. | 2017 | *Tacotron: Towards End-to-End Speech Synthesis* | Interspeech 2017 |
| [7] | Hyvärinen, A. | 2005 | *Estimation of Non-Normalized Statistical Models by Score Matching* | JMLR |
| [8] | Villani, C. | 2009 | *Optimal Transport: Old and New* | Springer |
| [9] | Albergo, M.S., Vanden-Eijnden, E. | 2022 | *Building Normalizing Flows with Stochastic Interpolants* | arXiv:2209.15571 |
| [10] | Lipman, Y., Chen, R.T.Q., et al. | 2022 | *Flow Matching for Generative Modeling* | ICLR 2023 |

---

## 📎 Phụ Lục — Giải Thích Thuật Ngữ

### Xác Suất Có Điều Kiện $p(x|y)$ Là Gì?

Là khả năng âm thanh $x$ xuất hiện **khi biết trước** văn bản $y$. Thay vì trả lời "xác suất bằng bao nhiêu?", nó trả lời câu hỏi: "Khi nói câu này, bao nhiêu khả năng giọng nói trông như thế này?"

> 🎙️ **Ví dụ**: Câu "Trời ơi!" có thể nói với giọng ngạc nhiên (70%), thất vọng (20%), hay đùa vui (10%). Hàm $p(x|y)$ thâu tóm tất cả khả năng đó.

### Tại Sao Không Dùng $L_2$ Loss Đơn Giản?

Nếu dùng $\|x_{\text{pred}} - x_{\text{real}}\|^2$, mô hình tối thiểu hóa bằng cách lấy **trung bình** của tất cả các âm thanh hợp lệ → giọng nói mờ nhạt, thiếu năng lượng (oversmoothed). Đây là lý do cần các phương pháp dựa trên phân phối (Diffusion, Flow Matching).

### Đa Tạp (Manifold) Là Gì?

Tưởng tượng không gian $\mathcal{X}$ là một căn phòng 6,880 chiều. Tiếng nói thực tế của con người không trải rộng khắp nơi trong phòng, mà chỉ nằm trên một "bề mặt mỏng" uốn lượn qua đó — gọi là đa tạp. Mô hình TTS cần học cách sinh sample **đúng trên bề mặt đó**.

### Flow Matching Loss — Đọc Như Thế Nào?

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\, x_t,\, \mathbf{y}}\, \bigl\|v_\theta(x_t, t, \mathbf{y}) - u_t(x_t \mid x_1)\bigr\|^2$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $v_\theta(x_t, t, \mathbf{y})$ | Hướng di chuyển mô hình **dự đoán** tại điểm $x_t$, bước $t$, với điều kiện text $\mathbf{y}$ |
| $u_t(x_t \mid x_1)$ | Hướng di chuyển **đúng** (ground truth từ đường thẳng nối noise $x_0$ với data $x_1$) |
| $\|\cdot\|^2$ | Sai số bình phương — phạt khi dự đoán sai hướng |

> 🚗 **Ví dụ**: Mô hình như học sinh lái xe, ground truth là giáo viên bên cạnh. Loss = độ sai lệch góc vô lăng. Training = luyện mãi cho đến khi luôn lái đúng hướng.
