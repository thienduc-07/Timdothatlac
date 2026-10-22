# BÁO CÁO XÁC MINH SỬA LỖI TÍCH HỢP (FIX VERIFICATION REPORT)
**Dự án:** Lost & Found AI Bus  
**Thời gian thực hiện:** 22/10/2026  
**Nội dung thực hiện:** Xử lý 2 lỗi tích hợp thực tế (Roboflow Model Version Switching & Twin-Seat Decomposition in Pipeline) + Khắc phục lỗi điều hướng giao diện.

---

## 1. TỔNG HỢP CÁC FILE ĐÃ SỬA

| File | Chức năng thay đổi |
| :--- | :--- |
| [`src/roboflow/roboflow_client.py`](file:///Users/thienduc/Desktop/Code/findlostitems/src/roboflow/roboflow_client.py) | Định tuyến suy luận theo đúng model version (`_submit_versioned`), xóa cache khi đổi version, theo dõi trạng thái `last_inference_info`, bảo vệ API key với `_mask_key`. |
| [`src/seats/seat_detector.py`](file:///Users/thienduc/Desktop/Code/findlostitems/src/seats/seat_detector.py) | Nối `TwinSeatDecomposer` vào `detect_seats_from_predictions` và `CustomSeatYOLOAdapter.detect`, tự động phân rã băng ghế rộng thành 2 ghế độc lập. |
| [`app.py`](file:///Users/thienduc/Desktop/Code/findlostitems/app.py) | Đọc `st.session_state['nav_to']` để kích hoạt trang sidebar tương ứng; hiển thị Model ID đích và cảnh báo lỗi suy luận trong Quản lý Model. |
| [`tests/test_audit_fixes_regression.py`](file:///Users/thienduc/Desktop/Code/findlostitems/tests/test_audit_fixes_regression.py) | Bổ sung 9 bài test hồi quy kiểm chứng Task 1 và Task 2. |
| [`docs/frontend_backend_audit.md`](file:///Users/thienduc/Desktop/Code/findlostitems/docs/frontend_backend_audit.md) | Tài liệu kiểm toán toàn diện frontend - backend và bảng kiểm kê 26 chức năng. |

---

## 2. NGUYÊN NHÂN GỐC VÀ HÀNH VI TRƯỚC / SAU KHI SỬA

### Lỗi 1: Roboflow Model Version Switching
- **Nguyên nhân gốc:**
  - Cloud Workflow `bus-seat-and-belongings-monitori` là Project Base Workflow trên Roboflow, được nền tảng bảo vệ chống sửa đổi qua API.
  - Bên trong workflow, step `roboflow_object_detection_model@v3` được Roboflow gán cố định `"model_id": "bus-seat-and-belongings-monitori/1"`.
  - Hàm `submit_image()` trước đây luôn gọi `_submit_sdk()` (`run_workflow`), khiến bất kể người dùng chọn Version nào trong giao diện, đám mây vẫn chỉ chạy Version 1.
- **Hành vi trước khi sửa:**
  - Chọn Version 2 trên giao diện -> request vẫn gửi tới workflow mang cấu hình Version 1 -> Không có cảnh báo, tạo ra thành công giả tạo.
- **Hành vi sau khi sửa:**
  - Khi chọn Version $V$: hệ thống định tuyến trực tiếp tới hosted model endpoint `https://detect.roboflow.com/{project_id}/{V}` (hoặc qua SDK với `model_id="{project_id}/{V}"`).
  - Request Version 1: Chạy thành công Version 1 (HTTP 200).
  - Request Version 2 (chưa train): Nhận mã lỗi thật `HTTP 404 Not Found`, ghi nhận trạng thái `FAILED` với thông báo lỗi đã được che API key (`KiM6****VLXM`), không tự ý fallback ngầm về Version 1.
  - Khi gọi `set_active_version()`, toàn bộ cache suy luận cũ lập tức bị xóa bỏ.

### Lỗi 2: Twin-Seat Decomposition trong Pipeline Phát hiện Ghế
- **Nguyên nhân gốc:**
  - Lớp `TwinSeatDecomposer` với thuật toán `split_horizontal_bench()` đã tồn tại nhưng hàm `detect_seats_from_predictions()` và `CustomSeatYOLOAdapter.detect()` chỉ nối thêm chuỗi `cluster_note = " (Cụm ghế đôi)"` và tạo **1 đối tượng `SeatConfig` duy nhất**.
- **Hành vi trước khi sửa:**
  - Băng ghế đôi (2 chỗ ngồi liền kề) chỉ có 1 bounding box và 1 Seat ID (`SEAT_001`). Khi 1 người ngồi bên trái, toàn bộ cụm ghế bị tính là có người, không thể gán đồ vật riêng cho ghế trống bên cạnh.
- **Hành vi sau khi sửa:**
  - Khi tỷ lệ `aspect_ratio > 1.45` và `width >= 0.14` hoặc nhãn là `double_seat`/`bench_seat`:
    Tự động gọi `TwinSeatDecomposer.split_horizontal_bench()`, sinh ra **2 ghế con riêng biệt** (ví dụ `SEAT_1A` và `SEAT_1B`) có khe hở giữa hai đệm ghế.
  - Kết hợp với `SeatTracker`, hai ghế con giữ nguyên ID qua các frame liên tiếp, không tạo thêm ghế thừa (zero duplication).
  - Trạng thái người ngồi (`SeatOccupancyEstimator`) được quản lý riêng cho từng ghế con.

### Lỗi 3: Nút Thao Tác Nhanh trên Dashboard không Chuyển Trang
- **Nguyên nhân gốc:**
  - Nút bấm gán `st.session_state['nav_to'] = "..."` rồi `st.rerun()`, nhưng widget `st.radio()` ở sidebar luôn khởi tạo với `index=0`, làm giao diện luôn quay lại trang Dashboard.
- **Hành vi sau khi sửa:**
  - Đọc `nav_target = st.session_state.pop("nav_to", None)` và gán index của trang tương ứng trong danh sách `page_list`. Bấm nút sẽ chuyển trang tức thì.

---

## 3. KẾT QUẢ KIỂM THỬ THỰC TẾ

### 1. Test đã chạy thật (100% PASS)
Lệnh chạy: `PYTHONPATH=. ./venv/bin/pytest -v`
```
======================= 140 passed, 16 warnings in 5.73s =======================
```
- 131 test cũ: Đều PASS, không bị xóa hay sửa đổi gian lận.
- 9 test mới:
  1. `test_roboflow_version_routing_target_model`: Xác minh chuyển version 1 và 2 thay đổi target model ID tương ứng.
  2. `test_roboflow_version_switching_cache_invalidation`: Xác minh xóa sạch cache khi chuyển version.
  3. `test_roboflow_version_2_non_existent_fails_without_fallback`: Giả lập version không tồn tại, xác minh trả về `None` và lưu lỗi 404 thật, không fallback ngầm.
  4. `test_roboflow_api_key_masked_in_exceptions`: Xác minh che toàn bộ ký tự nhạy cảm của API key thành `KiM6****VLXM`.
  5. `test_wide_bench_decomposes_into_two_child_seats`: Băng ghế đôi tạo đúng 2 ghế con có tọa độ và ID tách biệt.
  6. `test_single_seat_not_split`: Ghế đơn thông thường không bị cắt đôi sai lệch.
  7. `test_child_seats_geometry_within_parent_bounds`: Tọa độ 2 ghế con nằm hoàn toàn bên trong ranh giới cụm ghế mẹ.
  8. `test_child_seats_stable_ids_over_frames`: Chạy qua nhiều frame, 2 ghế con giữ ID ổn định tuyệt đối, không sinh duplicate.
  9. `test_child_seats_independent_occupancy_and_occlusion`: Người ngồi Ghế A làm Ghế A thành `OCCUPIED`, Ghế B vẫn giữ `VACANT`.

### 2. Test gửi Request thực tế tới Roboflow
- Request Version 1:
  - URL: `https://detect.roboflow.com/bus-seat-and-belongings-monitori/1`
  - Kết quả: `HTTP 200 OK`, `resolved_model: le-uc-s-workspace/bus-seat-and-belongings-monitori-1-rfdetr-small-t1`
- Request Version 2:
  - URL: `https://detect.roboflow.com/bus-seat-and-belongings-monitori/2`
  - Kết quả: `HTTP 404 Not Found` (xác thực hệ thống không bị lỗi ngầm, báo đúng thực tế).

---

## 4. NHỮNG THAO TÁC CẦN THỰC HIỆN TRÊN ROBOFLOW KHI MUỐN DÙNG VERSION 2
Do tài khoản Roboflow hiện có 58 ảnh chưa gán nhãn và chưa được train thành Version 2 trên cloud, người dùng cần thực hiện các bước sau trên Roboflow Web khi muốn sinh Version 2 thật:
1. Đăng nhập [app.roboflow.com](https://app.roboflow.com), vào dự án `bus-seat-and-belongings-monitori`.
2. Gán nhãn cho các ảnh còn lại trong mục *Annotate*.
3. Bấm *Generate Version* -> chọn kích thước ảnh và augmentations -> bấm *Generate* (miễn phí).
4. Bấm *Start Training* (sử dụng Roboflow training credit).
5. Khi training xong, quay lại tab **Quản lý Model** trên ứng dụng, chọn **Version 2** và bấm **Kích hoạt Phiên bản Model**. Hệ thống sẽ lập tức suy luận bằng weights mới của Version 2.

---

## 5. NHỮNG GÌ CHƯA THỂ XÁC MINH TRÊN CAMERA XE BUÝT THẬT
- **Camera trần xe buýt vật lý:** Do môi trường phát triển hiện tại chạy trên máy Mac / sandbox, chưa kết nối trực tiếp với camera IP/RTSP gắn trên trần xe buýt thật. Ứng dụng đã có sẵn luồng thu nhận tối ưu `ThreadedCameraReader` (Zero-Latency Buffer) sẵn sàng cắm là chạy.
- **Tài khoản Telegram & Email thật của người vận hành:** Cần cấu hình biến môi trường `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `SMTP_SERVER` trong file `.env` khi đưa vào vận hành tại bến xe.
