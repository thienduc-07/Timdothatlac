# 🚌 Lost & Found AI cho Xe Buýt (True Computer Vision & Occlusion-Aware Architecture)

Hệ thống Computer Vision thực thụ phân tích video từ camera gắn trên trần xe buýt nhìn thẳng xuống khoang hành khách (top-down view) để phát hiện vật dụng có nguy cơ bị bỏ quên.

Dự án tuân thủ nghiêm ngặt **nguyên tắc minh bạch kỹ thuật và Computer Vision thực chất**:
* **Không dùng lưới ghế giả lập làm cơ chế AI chính:** Lưới hình học chỉ đóng vai trò phương án dự phòng kiểm thử luồng (`GEOMETRY FALLBACK`), được công bố minh bạch và không mạo danh là AI detection.
* **Nhận diện ghế trực tiếp từ hình ảnh bằng Custom Deep Learning Model:** Tự động đọc danh sách class từ file weights, kiểm tra tính tương thích và từ chối lớp `chair` nhìn ngang của COCO 80.
* **Phân rã ghế đôi kề nhau trong cùng một hàng:** Hai ghế cạnh nhau luôn là 2 thực thể tách biệt (`SEAT_1A`, `SEAT_1B`), không gộp thành một vùng lớn.
* **Phân biệt độc lập 4 nhóm định danh (ID Separation):** Seat ID, Person Track ID, Object Track ID và Event ID.
* **Quy trình Huấn luyện Thật (Real Training Pipeline):** Chạy huấn luyện Ultralytics trong tiến trình nền, trích xuất metrics thật từ `results.csv`, loại bỏ hoàn toàn các hàm sinh chỉ số mAP giả lập.

---

## 📌 1. Bối cảnh & Yêu cầu Kỹ thuật Bắt buộc

1. **Phân biệt 4 nhóm ID không đồng nhất:**
   - `Seat ID` (ví dụ `SEAT_1A`, `SEAT_1B` hoặc `SEAT_001`): Định danh cố định gắn với vị trí ghế trên sàn xe buýt.
   - `Person Track ID` (ví dụ `P01`, `P02`): ID theo dõi hành khách qua các frame của tracker ByteTrack.
   - `Object Track ID` (ví dụ `O01`, `O02`): ID theo dõi đồ vật do tracker duy trì.
   - `Event ID` (ví dụ `EVT_25C0B567`): ID sự kiện vòng đời món đồ, giữ nguyên qua các chu kỳ che khuất và xuất hiện lại.
2. **Ước lượng trạng thái có người bằng vùng tiếp xúc cơ thể (Contact Zone):**
   - Không gán ghế chỉ bằng tâm bounding box của người (dễ lệch do góc camera trần hoặc đầu/vai nghiêng ra lối đi).
   - Phân tích vùng thân dưới/tiếp xúc đệm ghế (pelvic contact zone: 45%-50% nửa dưới bounding box người).
   - Nếu người ngồi lấn giữa 2 ghế hoặc nằm ngang cả 2 ghế, hệ thống trả về `AMBIGUOUS / UNKNOWN` thay vì ép buộc gán cho một ghế.
3. **Gán đồ vật cho ghế (Object-Seat Association):**
   - Đồ vật nằm ở ranh giới giữa 2 ghế -> gán `UNKNOWN_SEAT`.
   - Khi ghế đang có người ngồi, không cảnh báo bỏ quên chỉ vì phát hiện đồ vật tại đó.
   - Khi người đứng dậy rời ghế và đồ vật vẫn còn lại trên ghế trống, bắt đầu đánh giá nguy cơ (`ITEM_LEFT` ➔ `SUSPECTED`).
4. **Xử lý che khuất và bảo toàn sự kiện (Occlusion Handling):**
   - Khi người mới ngồi lên đè lên đồ vật: chuyển sang `OCCLUDED`, **giữ nguyên Event ID, timestamp và lịch sử**.
   - Khi người mới đứng dậy: nếu đồ vật xuất hiện lại, tái liên kết với sự kiện cũ; nếu không xuất hiện lại sau thời gian quy định, chuyển sang `REVIEW_REQUIRED` (yêu cầu nhân viên kiểm tra), **tuyệt đối không tự ý chuyển thành RESOLVED**.

---

## 🏗️ 2. Luồng Xử lý Dữ liệu Chuẩn Computer Vision

```
[ Video Frame ]
      │
      ▼
┌─────────────────────────────────┐
│ 1. Seat Detector                │ ◄─── Nạp weights models/bus_seat_yolo.pt
│ • Đọc class thực tế từ weights  │      (Nếu chưa có: báo CUSTOM SEAT MODEL NOT CONFIGURED,
│ • Từ chối COCO chair (nhìn ngang)     chỉ bật GEOMETRY FALLBACK khi người dùng chủ động cho phép)
│ • Phân tách ghế đôi (Twin Seats)│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 2. Person & Object Detector     │ ◄─── YOLOv8 (person, backpack, handbag, suitcase, bottle...)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 3. Trackers                     │
│ • Seat Tracker                  │ ─── Duy trì Seat ID vĩnh viễn, khử rung lắc (Spatial EMA),
│                                 │     bảo toàn cấu trúc ghế khi có người che (Seat Permanence)
│ • Object Tracker (ByteTrack)    │ ─── Gán Person Track ID (P..) và Object Track ID (O..)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 4. Seat Occupancy Estimator     │
│ • Vùng tiếp xúc thân dưới       │ ─── Phân định OCCUPIED / VACANT / AMBIGUOUS / UNKNOWN
│ • Twin-seat disambiguation      │ ─── Khử rung giật trạng thái bằng Temporal Hysteresis
│ • Departure Tracking            │ ─── Ghi nhận chính xác ai vừa rời ghế
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 5. Object-Seat Association      │ ─── Liên kết đồ vật với đệm ghế; ranh giới ➔ UNKNOWN_SEAT
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 6. Forgotten-Item Detector      │ ─── Đồ vật trên ghế trống ➔ ITEM_LEFT ➔ SUSPECTED
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 7. Occlusion Manager            │ ─── Người mới ngồi đè lên ➔ OCCLUDED (giữ nguyên Event ID)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 8. Event State Machine (DB)     │ ─── OBSERVED ➔ ITEM_LEFT ➔ SUSPECTED ➔ OCCLUDED ➔ RECHECK ➔ REVIEW_REQUIRED
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ 9. UI & Visualization           │
│ • Streamlit Dashboard           │ ─── Hiển thị rõ cơ chế: MODEL DETECTION / GEOMETRY FALLBACK
│ • VideoWriter Annotations       │ ─── Màu chuẩn: Seat (Cyan), Person (Green), Object (Amber), Alert (Red), Occluded (Purple)
└─────────────────────────────────┘
```

---

## 🔍 3. Minh bạch Cơ chế Nhận diện Ghế (Seat Detection Modes)

Hệ thống phân định rạch ròi 4 trạng thái nhận diện ghế trên toàn bộ giao diện:

| Chế độ | Mô tả | Nhãn giao diện |
| :--- | :--- | :--- |
| **MODEL DETECTION** | Nhận diện từng đệm ghế trực tiếp từ hình ảnh bằng mô hình Deep Learning chuyên biệt. | `AI MODEL DETECTION (Custom Deep Learning)` (Xanh lục) |
| **GEOMETRY FALLBACK** | Dự phòng tính toán tham số phục vụ kiểm thử luồng khi chưa nạp weights. Tuyệt đối không mạo nhận là AI. | `GEOMETRY FALLBACK (Dự phòng kiểm thử, KHÔNG PHẢI AI)` (Vàng cam cảnh báo) |
| **MANUAL CONFIGURATION** | Sơ đồ ghế do người quản trị thiết lập thủ công qua giao diện. | `MANUAL CONFIGURATION` (Xanh dương) |
| **NOT CONFIGURED** | Chưa nạp file weights và chưa bật tùy chọn dự phòng kiểm thử. | `CUSTOM SEAT MODEL NOT CONFIGURED` (Đỏ) |

> [!WARNING]
> **Tại sao lớp `chair` (ID 56) của YOLO COCO bị từ chối?**
> Tập dữ liệu COCO được huấn luyện trên ghế phòng khách và ghế văn phòng từ góc chụp ngang. Trong góc nhìn từ trần xe buýt hướng xuống, đệm ghế chỉ là bề mặt phẳng chữ nhật, hoàn toàn khác biệt với đặc trưng góc nhìn ngang của COCO. Việc dùng lớp `chair` của COCO cho camera trần xe buýt sẽ gây nhiễu và báo sai nghiêm trọng.

---

## 🚀 4. Quy trình Huấn luyện Thật (Real Training Pipeline)

Dự án cung cấp pipeline huấn luyện YOLO hoàn chỉnh thay thế toàn bộ logic giả lập trước đây:

1. **Pre-flight Dataset Validation:** Kiểm tra cấu trúc `data.yaml`, xác thực sự tồn tại của thư mục ảnh `train/images`, `val/images` và kiểm tra các file nhãn `.txt` không rỗng trước khi cho phép chạy huấn luyện.
2. **Kiểm tra phần cứng tự động:** Nhận diện GPU NVIDIA (CUDA), Apple Silicon GPU (MPS) hoặc CPU kèm khuyến cáo hiệu năng.
3. **Chạy huấn luyện trong tiến trình nền:** Sử dụng `subprocess.Popen` gọi `python -m ultralytics train`, không làm đơ giao diện Streamlit.
4. **Ghi nhật ký thời gian thực:** Ghi toàn bộ tiến độ vào `data/training_logs/{job_id}.log` và cho phép theo dõi trực tiếp trên UI qua Log Viewer.
5. **Trích xuất Metrics Thật:** Khi huấn luyện hoàn tất, đọc trực tiếp các chỉ số `precision`, `recall`, `mAP50`, `mAP50-95` từ file `results.csv` do Ultralytics sinh ra.
6. **Candidate Model Review & Approval:** Người quản trị phê duyệt phiên bản model và lưu vào SQLite database trước khi kích hoạt sang môi trường suy luận.

---

## 🧪 5. Kết quả Kiểm thử & Nghiệm thu (Test Coverage)

Toàn bộ **86/86 test cases** trong repository đều vượt qua (100% PASS), bao gồm file kiểm thử nghiệm thu chuyên biệt `tests/test_real_cv_and_training_audit.py` bao phủ 15 kịch bản bắt buộc:

1. `test_01_two_adjacent_seats_have_separate_seat_ids`: Hai ghế liền nhau có hai Seat ID riêng biệt (`SEAT_1A` và `SEAT_1B`), có khe hở đệm.
2. `test_02_seat_id_stability_when_detection_shifts_slightly`: Seat tracker duy trì ID vĩnh viễn khi detection rung lắc nhẹ.
3. `test_03_person_on_seat_a_seat_b_remains_vacant`: Người ngồi ở Ghế A thì chỉ Ghế A `OCCUPIED`, Ghế B vẫn `VACANT`.
4. `test_04_seat_b_has_object_but_no_person`: Ghế B có đồ vật nằm trên mặt đệm nhưng không có người.
5. `test_05_person_a_stands_up_while_person_b_remains_seated`: Người ở Ghế A đứng lên thì Ghế A chuyển về `VACANT`, Ghế B vẫn `OCCUPIED`.
6. `test_06_person_b_stands_up_and_object_remains`: Người ở Ghế B đứng lên và đồ vật còn lại -> chuyển `ITEM_LEFT` ➔ `SUSPECTED`.
7. `test_07_new_person_sits_on_object_causing_occlusion`: Người mới ngồi đè lên đồ vật -> chuyển `OCCLUDED`, giữ nguyên Event ID.
8. `test_08_person_stands_up_object_reappears_relinks_to_event`: Người mới đứng lên, đồ vật xuất hiện lại -> tái liên kết sự kiện cũ.
9. `test_09_person_stands_up_object_missing_review_required`: Người mới đứng lên nhưng đồ vật không xuất hiện lại -> chuyển `REVIEW_REQUIRED`, không tự ý `RESOLVED`.
10. `test_10_object_on_boundary_assigned_unknown_seat`: Đồ vật nằm sát ranh giới 2 ghế -> gán `UNKNOWN_SEAT`.
11. `test_11_model_without_seat_class_rejected`: Model không có lớp seat (COCO chair) bị từ chối.
12. `test_12_no_custom_weights_reports_not_configured`: Không có file weights -> báo rõ `CUSTOM SEAT MODEL NOT CONFIGURED`.
13. `test_13_dataset_lacking_annotations_fails_validation`: Dataset thiếu annotation bị từ chối trước khi train.
14. `test_14_training_failure_recorded_properly`: Tiến trình train thất bại được ghi nhận chính xác trạng thái `failed`.
15. `test_15_no_fake_metrics_generated_without_real_validation`: `simulate_training_job()` bị vô hiệu hóa, không tạo metrics giả.

### Lệnh chạy kiểm thử:
```bash
# Chạy toàn bộ 86 bài test
PYTHONPATH=. ./venv/bin/pytest -v

# Chạy riêng 15 kịch bản nghiệm thu kiểm toán
PYTHONPATH=. ./venv/bin/pytest tests/test_real_cv_and_training_audit.py -v
```

---

## 💻 6. Hướng dẫn Khởi chạy Ứng dụng

### Khởi động Web Dashboard:
```bash
./venv/bin/streamlit run app.py
```
Mở trình duyệt tại: `http://localhost:8501`.

### Hướng dẫn Nạp Custom Weights:
Để kích hoạt chế độ `MODEL DETECTION` thay vì `GEOMETRY FALLBACK`:
1. Chuẩn bị file trọng số YOLOv8 đã train trên dữ liệu xe buýt (các lớp: `bus_seat`, `seat`, `seat_window`, `seat_aisle`).
2. Đặt file trọng số vào thư mục: `models/bus_seat_yolo.pt`.
3. Hệ thống sẽ tự động đọc danh sách class từ metadata weights và chuyển sang chế độ AI nhận diện ghế thực tế.

---

## ⚠️ 7. Các Giới hạn Kỹ thuật Còn lại (Technical Limitations)

1. **Góc nghiêng và méo thấu kính méo góc rộng (Fisheye Distortion):**
   - Camera góc siêu rộng (140°-180°) trên trần xe buýt gây méo hình ở mép rìa khung hình. Đệm ghế ở rìa có hình thang cong thay vì chữ nhật phẳng. Cần bổ sung bước hiệu chuẩn thấu kính (Camera Undistortion matrix) trước khi inference nếu dùng camera fisheye thực tế.
2. **Vật thể kích thước nhỏ (Small Objects):**
   - Các vật mỏng và nhỏ như thẻ xe buýt, tai nghe bluetooth, ví da mỏng khi đặt trùng màu với đệm ghế có thể khó phát hiện từ độ phân giải 720p/1080p ở độ cao 2.2m-2.5m. Khuyến nghị sử dụng camera tối thiểu 2K hoặc camera phụ trợ.
3. **Trường hợp che khuất hoàn toàn và hành khách cầm nhầm:**
   - Nếu hành khách ngồi đè lên đồ vật rồi nhét vào ba lô cá nhân trước khi đứng dậy, camera không thể nhìn thấy bên trong ba lô. Hệ thống sẽ giữ nguyên tắc an toàn: chuyển sự kiện sang `REVIEW_REQUIRED` để người vận hành kiểm tra lại lịch sử video, không tự ý kết luận món đồ đã được lấy đúng chủ.
