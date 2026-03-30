# Tài Liệu Test Cases: Hệ Thống F5-TTS Studio

Tài liệu này liệt kê chi tiết các tình huống kiểm thử (Test Cases) dự tính áp dụng cho quá trình kiểm thử tự động trên giao diện Web TTS Studio (cổng `3001`).

| Mã TC | Chức năng (Feature) | Mục đích (Description) | Các bước thực hiện (Steps) | Kết quả mong muốn (Expected Result) |
| :---: | :--- | :--- | :--- | :--- |
| **TC01** | Điều hướng | Kiểm tra tải giao diện và cấu trúc chung | 1. Mô phỏng mở Chromium/Firefox.<br>2. Truy cập `http://localhost:3001` | - Trang web tải thành công hoàn toàn.<br>- Tiêu đề ứng dụng góc trái hiển thị "🎙️ F5-TTS Studio".<br>- Thanh Sidebar hiển thị đủ 3 chức năng điều hướng. |
| **TC02** | Điều hướng | Kiểm tra sự kiên kết của các tab chức năng | 1. Bấm vào tab "2. Training".<br>2. Bấm vào tab "3. Testing".<br>3. Bấm vào tab "1. Labeling". | Mỗi khi Tab mới được chọn, tiêu đề trên nội dung chính sẽ thay hình đổi dạng tương quan (`Train F5-TTS Model`, `Generate Audio (Testing)`, `Data Labeling with MRS`). |
| **TC03** | Labeling Tab | Tích hợp thành công giao diện thu âm chéo | 1. Ở trang chính, đi tới "1. Labeling".<br>2. Nội dung card chính hiển thị màn hình nhúng Mimic | Tìm thấy một thẻ `iframe` hợp lệ chỉ định tới đường dẫn URL chéo của container Mimic (`http://localhost:3000`). |
| **TC04** | Training Tab | Các trường input điền sẵn thông số an toàn | 1. Điều hướng tới "2. Training".<br>2. Kiểm tra những Value của trường Settings. | - Box Dataset có dữ liệu tuỳ chọn.<br>- Base Model mặc định đang giữ `F5TTS-Base`.<br>- Epoch mặc định chuẩn `10`.<br>- Batch Size chuẩn `4`. |
| **TC05** | Training Tab | Luồng UI sau khi gửi yêu cầu Huấn Luyện API | 1. Để mặc định cấu hình hoặc thiết lập lại tuỳ định.<br>2. Bấm "Start Training". | - Nút đổi tên là "Training in Progress..." kèm vòng xoay chờ.<br>- Nút bị Disabled để ngăn người dùng Double Request.<br>- Khung Console Log giả lập nảy sinh đoạn ký văn bản thông báo "Training initialized...". |
| **TC06** | Training Tab | Call API Dataset List từ hệ thống gốc API Python | 1. Đi tới "2. Training".<br>2. Quan sát Ô Dropdown danh sách. | Phải đọc được các tệp đang lưu trên `/datasets` của Container phụ hoặc đổ Auto Dummy string `vietnamese_train`. |
| **TC07** | Testing Tab | Trải nghiệm phím Generate cực nhanh từ câu mẫu | 1. Tham quan trang "3. Testing".<br>2. Nhìn vào mục vùng Textarea. | Textarea tự được Fill sẵn nội dung text (vd: Về thi hào Nguyễn Du) để tiết kiệm thời gian gõ test cho Dev. |
| **TC08** | Testing Tab | Ngăn chặn Bug gửi chuỗi Rỗng vào Server Inference | 1. Vào Tab "Testing".<br>2. Chọn trọn văn kiện ở Textarea và Delete/Backspace.<br>3. Quan sát nút bấm Generate. | Form input phản hồi lại việc bị trống dẫn đến Button "Generate Voice" bị block màu xám (Disabled). |
| **TC09** | Testing Tab | Xác minh quy trình Sinh Audio Khép Kín Full Flow | 1. Trở lại "Testing".<br>2. Bấm "Generate Voice" từ Input hợp lệ.<br>3. Đợi tiến trình Delay API giả lập hoặc thực chiến từ F5 Inference Server kết nối xuống UI. | - Nút chuyển trạng thái "Synthesizing...".<br>- Trực tiếp hiển thị một thanh Audio Player `.wav` trên hệ thống ngay khi Audio Endpoint HTTP(S) trả về. Người dùng có thể nghe thử. |

> Tài liệu này được sử dụng làm quy chiếu cho toàn bộ tập lệnh Test (`dashboard.spec.js` bằng thư viện Playwright) của kỹ sư chất lượng kiểm thử (QA).
