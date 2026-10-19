# 🚌 Lost & Found AI cho Xe Buýt (Top-Down Seat Detection & Occlusion-Aware Edition)

Hệ thống AI hỗ trợ phát hiện các vật dụng có khả năng bị bỏ quên trên xe buýt thông qua camera gắn trên trần xe hướng xuống các hàng ghế, tích hợp **module tự động nhận diện và phân biệt từng ghế riêng biệt**, **phân rã ghế đôi kề nhau trong cùng một hàng (Twin-Seat Decomposition)**, và **cơ chế nhận biết - xử lý tình huống che khuất (Occlusion-Aware)**.

---

## 📌 1. Bối cảnh & Yêu cầu Cốt lõi

Hệ thống được thiết kế chuyên biệt cho bối cảnh camera gắn trên trần xe buýt nhìn thẳng xuống khoang hành khách (top-down view):
1. **Không được gộp cả hàng ghế thành một vùng lớn duy nhất:** Mỗi ghế phải là một thực thể độc lập với Seat ID riêng (ví dụ `SEAT_1A`, `SEAT_1B`...).
2. **Không bắt buộc người dùng phải vẽ thủ công tất cả các ghế:** Hệ thống tự động quét, phát hiện và phân tách cấu trúc ghế từ frame camera.
3. **Phân rã ghế đôi liền kề trong cùng một hàng:** Băng ghế đôi được phân rã chính xác thành:
   - Ghế sát cửa sổ (Window seat - ví dụ `SEAT_1A`, `SEAT_1D`).
   - Ghế sát lối đi (Aisle seat - ví dụ `SEAT_1B`, `SEAT_1C`).
4. **Phân biệt độc lập 4 nhóm định danh (Không đồng nhất):**
   - `Seat ID`: Định danh cố định của chiếc ghế (ví dụ: `SEAT_1A`, `SEAT_1B`).
   - `Person Track ID`: Định danh tạm thời của hành khách qua các frame (ví dụ: `P01`, `P02`).
   - `Object Track ID`: Định danh tạm thời của món đồ (ví dụ: `O01`, `O02`).
   - `Event ID`: Định danh ổn định của sự kiện nghi ngờ bỏ quên (ví dụ: `EVT_8A7B2C1D`).
5. **Ước lượng trạng thái có người bằng vùng tiếp xúc cơ thể:** Không chỉ dựa vào tâm bounding box của người (dễ sai lệch khi góc nhìn nghiêng hoặc đầu/vai nghiêng sang lối đi); hệ thống đánh giá vùng tiếp xúc thân dưới (lower-body/pelvic zone) với diện tích đệm ghế.
6. **Xử lý người ngồi giữa 2 ghế kề nhau:** Nếu người ngồi lấn giữa 2 ghế hoặc nằm ngang cả 2 ghế, hệ thống đánh dấu trạng thái `AMBIGUOUS / UNKNOWN` thay vì ép buộc gán sai cho 1 ghế.
7. **Bảo toàn sự kiện khi bị che khuất (Occlusion Handling):** Khi có người mới ngồi xuống đè lên món đồ, sự kiện chuyển sang `OCCLUDED` và **giữ nguyên Event ID cùng toàn bộ lịch sử tích lũy**.

---

## 🏗️ 2. Kiến trúc Module Độc lập

```
                           [ Camera Frame / Video ]
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
     ┌──────────────────────┐                      ┌──────────────────────┐
     │ 1. Seat Detection    │                      │ 2. Object & Person   │
     │ (src/seats/detector) │                      │ Detection & Tracking │
     │ • Custom YOLO Engine │                      │ • YOLOv8 + ByteTrack │
     │ • Top-Down Vision    │                      │ • Person tracks: P.. │
     │ • TwinSeatDecomposer │                      │ • Object tracks: O.. │
     └──────────┬───────────┘                      └──────────┬───────────┘
                ▼                                             ▼
     ┌──────────────────────┐                      ┌──────────────────────┐
     │ 3. Seat Tracker      │                      │ 4. Contact Zone &    │
     │ (src/seats/tracker)  │                      │ Disambiguation       │
     │ • ID ổn định         │                      │ • Pelvic zone 50%    │
     │ • Spatial EMA filter │                      │ • Twin-seat overlap  │
     │ • Seat Permanence    │                      └──────────┬───────────┘
     └──────────┬───────────┘                                 │
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │ 5. Seat Occupancy Estimator   │
                       │ (src/seats/seat_occupancy)    │
                       │ • OCCUPIED / VACANT / UNKNOWN │
                       │ • Temporal Hysteresis filter  │
                       │ • Departure Tracking (Ai rời) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 6. Object-Seat Association    │
                       │ • Gán vật thể vào từng ghế    │
                       │ • Ranh giới ➔ UNKNOWN_SEAT    │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 7. Forgotten-Item & Occlusion │
                       │ (src/events/forgotten_item)   │
                       │ • Case A: Ghế có người che    │
                       │ • Case B/D: Mất trên ghế trống│
                       │ • Re-association bằng chứng   │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 8. Event Lifecycle (State)    │
                       │ (src/events/event_manager)    │
                       │ • State Machine (8 trạng thái)│
                       │ • Event ID cố định            │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 9. UI & Visualization         │
                       │ (Streamlit & VideoWriter)     │
                       │ • 5 lớp chú thích (Togglable) │
                       │ • Báo cáo số ghế & nghi ngờ   │
                       └───────────────────────────────┘
```

### Chi tiết các module trong `src/seats/`:
- `src/seats/seat_detector.py`:
  - `TwinSeatDecomposer`: Phân rã băng ghế đôi thành Ghế A (cửa sổ) và Ghế B (lối đi).
  - `CustomYOLOSeatDetector`: Adapter nạp trọng số YOLO chuyên biệt (`models/bus_seat_yolov8.pt`).
  - `TopDownVisionSeatDetector`: Thị giác máy tính nhận diện cấu trúc đối xứng khoang xe buýt.
  - `BusGeometrySeatDetector`: Tham số hóa hình học khoang xe theo số hàng và số ghế.
  - `SeatDetector.inspect_coco_classes()`: Kiểm tra minh bạch khả năng model; thông báo rõ ràng rằng YOLO COCO phổ thông chỉ có `chair` nhìn ngang, không giả lập AI.
- `src/seats/seat_tracker.py`:
  - Duy trì ánh xạ `seat_id` ổn định qua thời gian.
  - Bộ lọc Exponential Moving Average (EMA) triệt tiêu rung lắc tọa độ do động cơ xe rung.
  - **Cơ chế Seat Permanence:** Ghế gắn cố định trên sàn xe; khi hành khách ngồi đè lên che khuất đệm ghế, cấu trúc ghế vẫn được giữ nguyên vẹn và không bị xóa hay cấp ID mới.
- `src/seats/seat_occupancy.py`:
  - Trích xuất vùng tiếp xúc thân dưới của người (`y1 + 0.45*h` đến `y2`).
  - Phân xử ghế đôi: Nếu chênh lệch tỷ lệ giao nhau giữa 2 ghế nhỏ hơn 12% và cùng đè lên cả 2 ghế -> Đánh dấu `AMBIGUOUS / UNKNOWN`.
  - Ghi nhận người vừa rời khỏi ghế (`get_last_departed_person`).
- `src/seats/seat_manager.py`:
  - Tích hợp toàn diện Detector, Tracker và Occupancy Estimator.
  - Phân loại đồ vật: nếu nằm ở ranh giới giữa 2 ghế -> gán `UNKNOWN_SEAT`.

---

## 🧪 3. Kiểm thử Kịch bản Ghế đôi Liền kề (Scenario C)

Kịch bản kiểm thử cốt lõi được xây dựng trong `tests/test_twin_seats_and_occupancy.py`:

```
[ Frame 0-10 ]   Ghế 1A: Có người P1    │ Ghế 1B: Trống, có Chiếc ví O10
                 Trạng thái Ghế 1A: OCCUPIED  │ Trạng thái Ghế 1B: VACANT
                 ➔ Sự kiện ví O10: Bắt đầu theo dõi (ITEM_LEFT ➔ SUSPECTED: EVT_001)

[ Frame 11-17]   Người P1 ở Ghế 1A đứng lên rời đi. Ghế 1B vẫn có ví O10.
                 Trạng thái Ghế 1A: VACANT    │ Trạng thái Ghế 1B: VACANT
                 ➔ Sự kiện ví O10 trên Ghế 1B vẫn giữ nguyên EVT_001.

[ Frame 18-25]   Người mới P2 lên xe ngồi vào Ghế 1B (đè lên chiếc ví, ví mất detection).
                 Trạng thái Ghế 1B: OCCUPIED
                 ➔ Sự kiện ví O10 chuyển sang OCCLUDED; GIỮ NGUYÊN Event ID: EVT_001.

[ Frame 26-35]   Người mới P2 đứng lên rời khỏi Ghế 1B.
                 + Trường hợp 1 (Ví xuất hiện lại): Hệ thống tái liên kết bằng chứng vị trí,
                   khôi phục sự kiện EVT_001 về SUSPECTED.
                 + Trường hợp 2 (Ví biến mất): Ghế 1B vào RECHECK (0.5s-2s), sau đó
                   chuyển sang REVIEW_REQUIRED để nhân viên nhà xe kiểm tra.
```

---

## 🎨 4. Giao diện Giám sát & Bật/Tắt Lớp Chú thích (Layer Filter)

Hệ thống hiển thị trực quan trên Streamlit với bảng màu đồng nhất:
- **Ghế xe buýt (Cyan / Xanh dương):** Hiển thị `SEAT_1A`, `SEAT_1B` kèm trạng thái `[TRỐNG]`, `[CÓ NGƯỜI]`, `[CHƯA RÕ]`.
- **Hành khách (Green / Xanh lá):** Hiển thị `P01`, `P02` kèm độ tin cậy.
- **Đồ vật (Amber / Vàng cam):** Hiển thị `O10 (Ba lô)`, `O25 (Túi xách)`.
- **Sự kiện nghi ngờ (Red / Đỏ):** Viền đỏ nổi bật kèm nhãn `[⚠ NGHI NGỜ]`.
- **Vật thể bị che khuất (Purple / Tím):** Viền nét đứt kèm nhãn `[🛡 CHE KHUẤT]`.

### Bảng điều khiển thời gian thực:
Hiển thị đầy đủ số liệu:
- 🪑 **Tổng số ghế đã nhận diện**
- 👤 **Số ghế đang có người**
- 🟢 **Số ghế đang trống**
- 🎒 **Số đồ vật đang theo dõi**
- ⚠️ **Số sự kiện nghi ngờ bỏ quên**
- 🛡️ **Số sự kiện đang bị che khuất**

---

## 🚀 5. Hướng dẫn Chạy & Kiểm thử

### Bước 1: Kích hoạt môi trường ảo
```bash
cd /Users/thienduc/Desktop/Code/findlostitems
source venv/bin/activate
```

### Bước 2: Chạy toàn bộ 71 Unit Tests
```bash
PYTHONPATH=. ./venv/bin/pytest tests/ -v
```
*(100% 71/71 tests đã passed, bao gồm 10 tests kiểm tra chi tiết ghế đôi, tracking ổn định, disambiguation và kịch bản che khuất).*

### Bước 3: Khởi động Ứng dụng Streamlit
```bash
./venv/bin/streamlit run app.py
```
Mở trình duyệt tại: `http://localhost:8501`.

---

## 🧠 6. Hướng dẫn Huấn luyện Custom Overhead Bus Seat Model (Tùy chọn)

Nếu muốn nâng cấp từ cơ chế Top-Down Vision & Geometry Decomposition lên Deep Learning thuần túy:
1. Vào tab **AI Training & Dataset** trên giao diện web để thu thập 200-500 ảnh chụp từ trần xe buýt.
2. Gán nhãn 2 lớp: `seat_window` (ghế cửa sổ) và `seat_aisle` (ghế lối đi).
3. Xuất dataset chuẩn YOLOv8 `.zip`.
4. Train mô hình: `yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640`.
5. Lưu file trọng số vào: `models/bus_seat_yolov8.pt`. Hệ thống sẽ tự động ưu tiên nạp Custom Model này khi khởi động.
