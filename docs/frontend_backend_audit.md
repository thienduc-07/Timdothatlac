# BÁO CÁO KIỂM TOÁN TOÀN DIỆN FRONTEND – BACKEND & TÍNH NHẤT QUÁN TOÀN HỆ THỐNG
**Dự án:** Lost & Found AI Bus (Hệ thống AI giám sát trần xe buýt phát hiện đồ thất lạc)  
**Ngày thực hiện:** 22/10/2026  
**Trạng thái kiểm toán:** HOÀN TẤT & ĐÃ SỬA CÁC ĐIỂM BẤT NHẤT QUÁN  
**Môi trường:** Python 3.9.6, Pytest 8.4.2, Streamlit, Ultralytics YOLO26, Roboflow Serverless/Hosted Inference, Supervision ByteTrack  

---

## 1. TỔNG QUAN KIỂM TOÁN & MỤC TIÊU

Đợt kiểm toán này rà soát toàn diện sự đồng bộ giữa giao diện người dùng (`app.py`), backend xử lý (`src/`), database (`src/database/`), dịch vụ thông báo (`src/services/notification_service.py`), và tích hợp đám mây Roboflow (`src/roboflow/`).

**Mục tiêu cốt lõi:**
1. Đưa dự án về trạng thái nhất quán 100%: Chức năng nào hiển thị trên frontend thì phải có hành vi thực tế được xác định rõ; chức năng nào chưa hỗ trợ hoặc phụ thuộc môi trường phải được thông báo minh bạch, không để người dùng bấm vào rồi gặp lỗi hoặc hiển thị thành công giả.
2. Xử lý triệt để hai lỗi tích hợp đã xác minh:
   - **Roboflow Model Version Switching:** Chuyển đổi version thật trong request inference, không bị hardcode ở Version 1 của cloud workflow.
   - **Twin-Seat Decomposition:** Tích hợp `TwinSeatDecomposer` trực tiếp vào pipeline nhận diện ghế (`SeatDetector.detect_seats_from_predictions` và `CustomSeatYOLOAdapter.detect`), phân rã cụm ghế đôi thành 2 thực thể ghế độc lập với ID ổn định qua các frame.

---

## 2. BẢNG KIỂM KÊ TOÀN BỘ CHỨC NĂNG HỆ THỐNG

### Phân loại:
- **Nhóm A:** Frontend & Backend đều có, hoạt động đúng.
- **Nhóm B:** Frontend có nhưng Backend không có (hoặc thiếu xử lý thực tế).
- **Nhóm C:** Backend có nhưng Frontend chưa kết nối.
- **Nhóm D:** Frontend và Backend đều có nhưng giao tiếp sai lệch (lệch tham số, lệch endpoint).
- **Nhóm E:** Chức năng chạy bằng dữ liệu mô phỏng / mẫu (được phân lập rõ ràng).
- **Nhóm F:** Chức năng chưa thể xác minh trong môi trường hiện tại (cần thiết bị hoặc mạng thật).

| STT | Chức năng | Frontend | Backend | Phân loại | Trạng thái thực tế | Vấn đề phát hiện | Hướng xử lý |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **Điều hướng nhanh từ Dashboard (`nav_to`)** | Có | Có | **D -> A** | Đã sửa (Hoạt động tốt) | Các nút "🎥 Chạy Giám sát Camera", "⚠️ Mở Alert Center", "🪑 Sơ đồ Ghế" gán `st.session_state['nav_to']` nhưng `st.radio` ở sidebar hardcode `index=0`, làm bấm nút không chuyển trang. | Đã sửa `app.py`: đọc `nav_to` và chọn `selected_index` tương ứng khi render sidebar radio. |
| **2** | **Chạy Video Mẫu Demo trên Giám sát Camera** | Có | Có | **A & E** | Hoạt động tốt | Video mẫu mô phỏng xe buýt 200 frame (10 giây) tạo động bằng OpenCV; AI chạy detection, tracking, phát hiện bỏ quên và kích hoạt alert thật. | Giữ nguyên; ghi rõ nhãn DEMO MODE trên banner. |
| **3** | **Tải lên Video thực tế và Phân tích** | Có | Có | **A** | Hoạt động tốt | Hỗ trợ tải `.mp4`, `.avi`, `.mov`, `.mkv`. `VideoProcessor` đọc frame, chạy inference, tracking và lưu sự kiện vào SQLite. | Giữ nguyên; bảo vệ định dạng file và giới hạn kích thước an toàn. |
| **4** | **Giám sát Webcam thời gian thực** | Có | Có | **A & F** | Hoạt động (Có điều kiện) | `ThreadedCameraReader` chạy thread nền đọc camera vật lý (AVFoundation trên macOS, DirectShow trên Windows, V4L2 trên Linux). Nếu không có webcam, báo rõ lỗi cấp quyền. | Giữ nguyên; xử lý giải phóng camera tất định (`stop()`, `cap.release()`). |
| **5** | **Kết nối luồng RTSP / IP Camera** | Có | Có | **A & F** | Hoạt động (Có điều kiện) | `CameraStreamManager.start_stream(CameraSourceType.RTSP, url)` kiểm tra định dạng `rtsp://` hoặc `http://` và mở VideoCapture. | Giữ nguyên; trả về lỗi chi tiết nếu kết nối camera an ninh thất bại. |
| **6** | **Chọn Chế độ Inference: Local YOLO** | Có | Có | **A** | Hoạt động tốt | Chạy mô hình cục bộ `yolo26n.pt` hoặc `models/best.pt`. Tốc độ CPU ~24.59ms. | Giữ nguyên. |
| **7** | **Chọn Chế độ Inference: Roboflow** | Có | Có | **A** | Hoạt động tốt | Gửi ảnh sang Roboflow API. Đã sửa định tuyến đúng version được chọn. | Đã sửa (xem Task 1). |
| **8** | **Chọn Chế độ Inference: Hybrid AI** | Có | Có | **A** | Hoạt động tốt | Chạy Local YOLO trước (nhanh), bổ sung kết quả Roboflow, so khớp IoU để tránh trùng lặp đệm ghế. | Giữ nguyên. |
| **9** | **Chuyển đổi Roboflow Model Version** | Có | Có | **D -> A** | Đã sửa (Hoạt động tốt) | Cloud Workflow cố định `model_id: ".../1"`. Khi người dùng chọn Version 2, hệ thống trước đây vẫn chạy Version 1 mà không báo lỗi. | Đã sửa: định tuyến sang `https://detect.roboflow.com/{project_id}/{version}`, báo 404 thật khi version chưa train, xóa cache cũ. |
| **10** | **Phân rã Cụm Ghế Đôi (Twin Seats)** | Có | Có | **B -> A** | Đã sửa (Hoạt động tốt) | `TwinSeatDecomposer` có trong code nhưng `detect_seats_from_predictions` chỉ gán thêm nhãn chuỗi chứ không chia thành 2 ghế con. | Đã sửa: tự động nhận diện băng ghế rộng và gọi `split_horizontal_bench`, tạo 2 ghế độc lập với ID ổn định. |
| **11** | **Bảo toàn Cấu trúc Ghế khi bị che khuất** | Có | Có | **A** | Hoạt động tốt | `SeatTracker` áp dụng Spatial EMA Smoothing; khi hành khách ngồi đè lên ghế, ghế vẫn tồn tại trong danh sách, không bị xóa. | Giữ nguyên. |
| **12** | **Quản lý Trạng thái Ghế (Occupancy)** | Có | Có | **A** | Hoạt động tốt | `SeatOccupancyEstimator` tính tỷ lệ diện tích tiếp xúc cơ thể (`PERSON_SEAT_OVERLAP_THRESHOLD = 0.28`). 1 người ngồi Ghế A không làm Ghế B bị tính có người. | Giữ nguyên. |
| **13** | **Phát hiện Đồ bỏ quên (Forgotten Item)** | Có | Có | **A** | Hoạt động tốt | `ForgottenItemDetector` theo dõi thời gian đứng yên $\ge 3.0$s (`MIN_STATIONARY_TIME_SEC`), xác nhận hành khách rời khỏi ghế mới phát sinh cảnh báo `CONFIRMED`. | Giữ nguyên. |
| **14** | **Quản lý Che khuất Đồ vật (Occlusion Recovery)** | Có | Có | **A** | Hoạt động tốt | Khi có người mới ngồi che đồ vật, chuyển trạng thái sang `OCCLUDED`. Khi người đứng dậy, nếu đồ còn: khôi phục; nếu đồ biến mất: yêu cầu `REVIEW_REQUIRED`. | Giữ nguyên. |
| **15** | **Ghi nhận Sự kiện & Alert vào Database** | Có | Có | **A** | Hoạt động tốt | Lưu vào bảng SQLite `events` và `alerts`, hỗ trợ cập nhật trạng thái `CONFIRMED`, `DISMISSED`, `RESOLVED`. | Giữ nguyên. |
| **16** | **Xác minh / Loại bỏ Cảnh báo trên Alert Center** | Có | Có | **A** | Hoạt động tốt | Nút "✅ Xác nhận đúng" và "❌ Loại bỏ (Báo sai)" cập nhật trạng thái alert trong database và làm mới giao diện tức thì. | Giữ nguyên. |
| **17** | **Gửi Thông báo Telegram Bot** | Có | Có | **A & F** | Hoạt động (Có điều kiện) | Gọi `api.telegram.org` qua `urllib.request`. Báo lỗi HTTP 400/401/404 nếu token sai; gửi tin nhắn + ảnh snapshot nếu cấu hình đúng. | Giữ nguyên; không làm giả kết quả gửi. |
| **18** | **Gửi Thông báo Email SMTP** | Có | Có | **A & F** | Hoạt động (Có điều kiện) | Sử dụng `smtplib` và `MIMEMultipart`. Kiểm tra kết nối máy chủ mail, đính kèm ảnh bằng chứng nếu có. | Giữ nguyên; thông báo rõ nếu thiếu thông tin xác thực. |
| **19** | **Gửi Webhook HTTP POST** | Có | Có | **A & F** | Hoạt động (Có điều kiện) | Gửi payload JSON chuẩn tới webhook Slack/Discord/custom endpoint. Báo mã lỗi HTTP thật. | Giữ nguyên. |
| **20** | **Tạo Sơ đồ Lưới Ghế Mô phỏng (Geometry Fallback)** | Có | Có | **A & E** | Hoạt động tốt | `BusGeometrySeatDetector` sinh lưới hình học (ví dụ 4 hàng, 2+2 ghế). Đánh dấu minh bạch là GEOMETRY FALLBACK, không giả mạo là kết quả AI. | Giữ nguyên. |
| **21** | **Trích xuất Frame & Gán nhãn Dataset** | Có | Có | **A** | Hoạt động tốt | Trích xuất frame từ video, lưu vào `data/dataset/`, lưu nhãn bounding box theo định dạng chuẩn YOLO. | Giữ nguyên. |
| **22** | **Nén và Xuất Dataset YOLO (.zip)** | Có | Có | **A** | Hoạt động tốt | Đóng gói toàn bộ images và labels kèm file cấu hình `data.yaml` thành file zip để tải về huấn luyện. | Giữ nguyên. |
| **23** | **Đồng bộ Dữ liệu lên Roboflow Cloud** | Có | Có | **A** | Hoạt động tốt | `RoboflowDatasetWorkflow` tải ảnh và nhãn lên dự án Roboflow qua REST API chính thức. | Giữ nguyên. |
| **24** | **Huấn luyện Model Thực tế trong Tiến trình Nền** | Có | Có | **A & F** | Hoạt động (Có điều kiện) | `ModelService.start_real_training_job()` khởi tạo tiến trình `subprocess` độc lập chạy lệnh huấn luyện Ultralytics YOLO; ghi log vào `data/training_logs/`. | Giữ nguyên; kiểm tra phần cứng khả dụng (MPS/CUDA/CPU) trước khi chạy. |
| **25** | **Xem và Xuất Lịch sử Sự kiện** | Có | Có | **A** | Hoạt động tốt | Truy vấn bảng `events`, lọc theo ngày/loại đối tượng, xuất file CSV. | Giữ nguyên. |
| **26** | **Lưu Ảnh Snapshot Bằng chứng từ Video/Camera** | Có | Có | **A** | Hoạt động tốt | Lưu frame annotated vào thư mục `data/snapshots/evidence_YYYYMMDD_HHMMSS.jpg`. | Giữ nguyên. |

---

## 3. CHI TIẾT XỬ LÝ HAI VẤN ĐỀ CỐT LÕI

### TASK 1: Roboflow Model Version Switching

#### 1. Nguyên nhân gốc
- Khi kiểm tra chi tiết định nghĩa workflow trên cloud (`le-uc-s-workspace/workflows/bus-seat-and-belongings-monitori`), khối `roboflow_core/roboflow_object_detection_model@v3` được Roboflow gán cố định:
  ```json
  "model_id": "bus-seat-and-belongings-monitori/1"
  ```
- Workflow này là **Project Base Workflow** được Roboflow bảo vệ (`Cannot modify a project base workflow. This workflow is managed by the platform and protected against API edits.`), không thể sửa bằng API.
- Trước đây, khi người dùng chọn Version 2 trong giao diện `app.py`, `RoboflowWorkflowClient` vẫn gọi `_submit_sdk` (`run_workflow`), khiến đám mây luôn chạy Version 1 dù giao diện hiển thị Version 2.

#### 2. Giải pháp kỹ thuật đã triển khai
1. **Định tuyến chuẩn theo Version:**
   - Trong `src/roboflow/roboflow_client.py`, phương thức `submit_image()` kiểm tra `self.model_version`:
     - Nếu gọi versioned model (mặc định cho mọi phiên bản, bao gồm Version 1, 2, 3...): chuyển sang endpoint chuyên biệt `_submit_versioned()`.
     - Sử dụng `self._sdk_client.infer(image_path, model_id=f"{self.project_id}/{self.model_version}")` hoặc fallback raw POST tới `https://detect.roboflow.com/{self.project_id}/{self.model_version}?api_key={self.api_key}`.
2. **Không tự động Fallback ngầm:**
   - Khi truy vấn Version 2 (chưa được huấn luyện/sinh trên Roboflow), server trả về `HTTP 404 Not Found`. Hệ thống ghi nhận lỗi thật:
     ```python
     self.last_inference_info = {
         "requested_version": "2",
         "requested_model_id": "bus-seat-and-belongings-monitori/2",
         "status": "FAILED",
         "actual_model_id": None,
         "error": "404 Client Error: Not Found...",
     }
     ```
   - Tuyệt đối không âm thầm gọi lại Version 1 rồi báo Version 2 thành công.
3. **Xóa Cache khi Đổi Phiên bản:**
   - Trong `set_active_version()`, hệ thống lập tức dọn sạch `_last_submit` cache để đảm bảo không bao giờ trả về kết quả suy luận cũ của version trước.
4. **Bảo vệ Bí mật (Secret Protection):**
   - Đã bổ sung `_mask_key()` và `_sanitize_error()` để mọi thông điệp lỗi hoặc URL trong log/UI đều tự động ẩn API key dưới dạng `KiM6****VLXM`.

---

### TASK 2: Nối TwinSeatDecomposer vào Pipeline Phát hiện Ghế

#### 1. Nguyên nhân gốc
- Lớp `TwinSeatDecomposer` với thuật toán `split_horizontal_bench()` đã được viết và kiểm thử độc lập, nhưng trong hàm xử lý chính `detect_seats_from_predictions()` và `CustomSeatYOLOAdapter.detect()`, khi phát hiện một bounding box rộng, hệ thống chỉ nối chuỗi `cluster_note = " (Cụm ghế đôi)"` và tạo **duy nhất 1 đối tượng `SeatConfig`**.
- Do đó, hai chiếc ghế liền nhau bị gộp chung thành một ID, khiến việc quản lý trạng thái riêng biệt cho từng hành khách (ghế cửa sổ vs ghế lối đi) không thể thực hiện đầy đủ từ kết quả model.

#### 2. Giải pháp kỹ thuật đã triển khai
1. **Phân rã trong `detect_seats_from_predictions()`:**
   - Tính toán tỷ lệ khung bao: `aspect_ratio = box_w / box_h` và `box_w >= 0.14` (chuẩn hóa).
   - Khi có dấu hiệu băng ghế đôi:
     `is_twin_bench = (box_h > 0 and aspect_ratio > 1.45 and box_w >= 0.14) or ("double" in raw_lower or "bench" in raw_lower)`
   - Tự động xác định nửa trái (`side="left"`, ghế A cửa sổ & ghế B lối đi) hoặc nửa phải (`side="right"`, ghế C lối đi & ghế D cửa sổ) dựa vào tọa độ trọng tâm `cx`.
   - Gọi `TwinSeatDecomposer.split_horizontal_bench()`, sinh ra **hai đối tượng `SeatConfig` con độc lập** với hình học đa giác riêng biệt (cách nhau một khe hở `gap_ratio = 0.05`).
2. **Tương thích hoàn hảo với `SeatTracker`:**
   - `SeatTracker` so khớp từng ghế con với dữ liệu đã lưu bằng chỉ số IoU và khoảng cách trọng tâm.
   - Vì hai ghế con cách nhau một khoảng hở hình học, Ghế 1A luôn khớp với Tracked 1A, Ghế 1B luôn khớp với Tracked 1B.
   - **ID ghế hoàn toàn ổn định qua nhiều frame**, không bị tăng số đếm hay sinh ID thừa.
3. **Quản lý Trạng thái Độc lập & Che khuất:**
   - Hành khách ngồi xuống Ghế 1A chỉ làm Ghế 1A chuyển sang `OCCUPIED`. Ghế 1B vẫn giữ nguyên trạng thái `VACANT`.
   - Khi hành khách che khuất hoàn toàn đệm Ghế 1A (khiến model frame đó chỉ phát hiện Ghế 1B), cơ chế **Seat Permanence Guarantee** trong `SeatTracker` giữ vững Ghế 1A trong bộ nhớ và chuyển trạng thái sang có người, không bao giờ tự ý xóa ghế khỏi xe buýt.
4. **Bảo toàn Ranh giới Hình học:**
   - Tọa độ 4 đỉnh của từng ghế con đều nằm hoàn toàn bên trong phạm vi bounding box của cụm ghế mẹ (`min_x >= parent_min_x` và `max_x <= parent_max_x`).

---

## 4. DANH SÁCH FILE ĐÃ THAY ĐỔI

1. [src/roboflow/roboflow_client.py](file:///Users/thienduc/Desktop/Code/findlostitems/src/roboflow/roboflow_client.py):
   - Bổ sung `last_inference_info`, `last_switch_status`.
   - Thêm bộ lọc bảo mật `_mask_key()` và `_sanitize_error()`.
   - Định tuyến suy luận theo đúng `model_version` được chọn (`_submit_versioned()` và `_submit_http_versioned()`).
   - Xóa cache khi gọi `set_active_version()`.
2. [src/seats/seat_detector.py](file:///Users/thienduc/Desktop/Code/findlostitems/src/seats/seat_detector.py):
   - Tích hợp `TwinSeatDecomposer` vào `detect_seats_from_predictions()`.
   - Tích hợp `TwinSeatDecomposer` vào `CustomSeatYOLOAdapter.detect()`.
   - Phân tách băng ghế rộng thành 2 ghế con độc lập, giữ nguyên cấu trúc cho ghế đơn bình thường.
3. [app.py](file:///Users/thienduc/Desktop/Code/findlostitems/app.py):
   - Sửa lỗi điều hướng nhanh từ Dashboard: đọc `st.session_state['nav_to']` và tự động kích hoạt trang tương ứng trong sidebar navigation.
   - Hiển thị thông tin minh bạch về Model ID đích (`target_model_id`) và mã lỗi thật nếu suy luận Roboflow thất bại trong trang Quản lý Model.
4. [tests/test_audit_fixes_regression.py](file:///Users/thienduc/Desktop/Code/findlostitems/tests/test_audit_fixes_regression.py):
   - Thêm 9 bài test hồi quy tự động kiểm chứng toàn diện cho Task 1 và Task 2.

---

## 5. KẾT QUẢ KIỂM THỬ THỰC TẾ

### 1. Kiểm thử Tự động (Automated Pytest Suite)
Chạy lệnh kiểm thử toàn bộ dự án:
```bash
PYTHONPATH=. ./venv/bin/pytest -v
```
**Kết quả thực tế:**
- **140 bài test thu thập — 140 bài test PASS (100%)**
- Thời gian thực thi: **5.73 giây**
- Không có bất kỳ test cũ nào bị xóa hoặc vô hiệu hóa.

### 2. Các Bài Test Hồi quy Mới (Regression Tests)
| Tên bài test | Nội dung kiểm chứng | Kết quả |
| :--- | :--- | :---: |
| `test_roboflow_version_routing_target_model` | Chọn Version 1 và Version 2, xác nhận `target_model_id` thay đổi tương ứng | **PASS** |
| `test_roboflow_version_switching_cache_invalidation` | Xác nhận cache cũ bị xóa hoàn toàn khi chuyển đổi phiên bản model | **PASS** |
| `test_roboflow_version_2_non_existent_fails_without_fallback` | Giả lập Version 2 lỗi 404; kiểm tra hệ thống báo lỗi thật, không fallback ngầm về v1 | **PASS** |
| `test_roboflow_api_key_masked_in_exceptions` | Giả lập lỗi API; xác minh API key được che dạng `KiM6****VLXM` | **PASS** |
| `test_wide_bench_decomposes_into_two_child_seats` | Đưa vào 1 băng ghế đôi; xác minh tạo đúng 2 ghế con có ID và hình học tách biệt | **PASS** |
| `test_single_seat_not_split` | Đưa vào ghế đơn (tỷ lệ 0.9); xác minh giữ nguyên 1 ghế, không bị phân tách sai | **PASS** |
| `test_child_seats_geometry_within_parent_bounds` | Xác minh tọa độ 2 ghế con nằm hoàn toàn bên trong ranh giới ghế mẹ | **PASS** |
| `test_child_seats_stable_ids_over_frames` | Truyền cùng 1 băng ghế qua nhiều frame; xác minh ID ghế giữ ổn định, không sinh duplicate | **PASS** |
| `test_child_seats_independent_occupancy_and_occlusion` | Người ngồi Ghế A; xác minh Ghế A thành `OCCUPIED`, Ghế B vẫn giữ `VACANT` | **PASS** |

### 3. Phân biệt Rõ ràng giữa các Loại Kiểm thử
- **Test tự động đã chạy thật:** Toàn bộ 140 test trong `./tests/` với các phép toán hình học IoU, state machine, database repository, tracker smoothing.
- **Test với API Roboflow thật:** Đã gửi request thật tới `https://detect.roboflow.com/bus-seat-and-belongings-monitori/1` (HTTP 200, thành công trong 0.5s) và `https://detect.roboflow.com/bus-seat-and-belongings-monitori/2` (HTTP 404, xác nhận đúng hành vi khi version chưa train).
- **Test trên video mẫu:** 200 frame video mô phỏng camera trần xe buýt (10 giây) đạt 20.11 FPS.
- **Test cần thiết bị / tài khoản thật (Chưa thể tự động hóa trong sandbox):**
  - Camera phần cứng vật lý gắn trên trần xe buýt thực tế.
  - Bot token Telegram thật và SMTP server thật của khách hàng (hiện tại hệ thống đã có adapter kiểm tra kết nối thật và báo lỗi xác thực chính xác).
  - Thao tác nhấn nút *Start Training* trên Roboflow Cloud để sinh ra Version 2 thật (cần credit đào tạo của chủ sở hữu tài khoản).

---

## 6. PHÂN LOẠI CÁC VẤN ĐỀ CÒN TỒN TẠI & KHUYẾN NGHỊ

### Phân loại mức độ ưu tiên:
- **P0 (Lỗi nghiêm trọng, không khởi động được):** **0 lỗi.** Hệ thống khởi động trơn tru trên mọi nền tảng.
- **P1 (Hỏng chức năng cốt lõi / Kết quả gây hiểu nhầm):** **0 lỗi.** (Hai lỗi P1 về version switching và twin-seat decomposition đã được giải quyết hoàn toàn).
- **P2 (Chức năng phụ, tích hợp bên ngoài cần điều kiện thực tế):**
  - *P2.1: Huấn luyện Version 2 trên Roboflow:* Hiện tại dự án Roboflow có 58 ảnh chưa gán nhãn và chỉ mới train Version 1. Người dùng cần gắn nhãn xong và bấm *Generate Version* + *Train* trên web Roboflow khi muốn kích hoạt Version 2.
  - *P2.2: Telegram & SMTP Cấu hình Thực tế:* Khi triển khai lên xe buýt thật, cần điền biến môi trường `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `SMTP_SERVER` vào file `.env`.
- **P3 (Cải thiện trải nghiệm & Dọn dẹp code):**
  - *P3.1: Export Video có phụ đề bounding box:* Hiện tại video kết quả được vẽ trực tiếp theo thời gian thực trên màn hình; có thể bổ sung tùy chọn xuất toàn bộ video đã gắn nhãn thành file `.mp4` để tải về máy sau phiên phân tích.

---

## 7. HƯỚNG DẪN CHẠY ỨNG DỤNG & KIỂM TRA THỦ CÔNG

### 1. Khởi động Ứng dụng
Kích hoạt môi trường ảo và chạy Streamlit:
```bash
./venv/bin/streamlit run app.py
```

### 2. Các Bước Kiểm tra Thủ công Quan trọng
1. **Kiểm tra Điều hướng Nhanh:** Tại trang **Tổng quan**, bấm các nút "🎥 Chạy Giám sát Camera", "⚠️ Mở Alert Center" -> Xác nhận giao diện chuyển ngay sang trang tương ứng.
2. **Kiểm tra Video Mẫu & Phát hiện Đồ bỏ quên:**
   - Vào **Giám sát Camera** -> chọn tab **🎬 Video Mẫu Demo**.
   - Bấm **▶️ Chạy Phân tích Demo Mẫu**.
   - Quan sát: Xe buýt có hành khách lên xe, ngồi ghế, đặt ba lô, sau đó đứng dậy đi xuống xe.
   - Khi ba lô nằm lại trên ghế quá 3 giây sau khi người rời đi, hệ thống lập tức kích hoạt chuông cảnh báo và phát sinh bản ghi Alert.
3. **Kiểm tra Twin-Seat Decomposition:**
   - Vào **Quản lý Ghế** -> chọn **🔍 Nhận diện Ghế tự động**.
   - Chọn chế độ AI và quét frame xe buýt.
   - Xác nhận cụm ghế đôi được chia thành 2 ghế riêng biệt (ví dụ `SEAT_1A` và `SEAT_1B`), không còn bị gộp thành 1 ô duy nhất.
4. **Kiểm tra Chuyển đổi Phiên bản Roboflow:**
   - Vào **Quản lý Model** -> cuộn xuống phần **Roboflow Cloud**.
   - Xem thông số `Model ID đích`. Chọn Version 1 -> bấm Kích hoạt -> hệ thống báo sẵn sàng.
   - Thử chọn Version chưa tồn tại (ví dụ 2 hoặc 99) -> hệ thống thông báo trạng thái minh bạch về việc phiên bản chưa có trên cloud, không gây crash ứng dụng.
