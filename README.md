# 🚌 Lost & Found AI cho Xe Buýt (Occlusion-Aware Edition)

Hệ thống AI hỗ trợ phát hiện các vật dụng có khả năng bị bỏ quên trên xe buýt thông qua camera gắn trên trần xe hướng xuống các hàng ghế, tích hợp **cơ chế nhận biết và xử lý tình huống che khuất (Occlusion-Aware)**.

---

## 📌 1. Bối cảnh & Tình huống Nghiệp vụ Bắt buộc

Trong thực tế, camera trần xe buýt thường gặp tình huống sau:
1. Hành khách $A$ ngồi trên ghế cùng một chiếc ví (hoặc túi xách, ba lô).
2. Hành khách $A$ đứng lên rời đi, nhưng chiếc ví vẫn nằm trên ghế.
3. Ghế được xác nhận là trống, hệ thống đếm thời gian và tạo trạng thái nghi ngờ (`SUSPECTED`) khi đủ điều kiện.
4. Một hành khách $B$ bước tới ngồi xuống ghế và **che khuất chiếc ví**.
5. Mô hình YOLO không còn phát hiện chiếc ví trong một khoảng thời gian.
6. **Yêu cầu cốt lõi:** Hệ thống **tuyệt đối không được xóa sự kiện hoặc kết luận chiếc ví đã biến mất** chỉ vì detection tạm thời không còn xuất hiện.
7. Khi hành khách $B$ đứng dậy (ghế quan sát được trở lại), hệ thống kiểm tra xem chiếc ví có xuất hiện trở lại hay không:
   - Nếu xuất hiện trở lại với các bằng chứng phù hợp (cùng ghế, vị trí tương đối, cùng loại vật thể): **Nối lại sự kiện cũ**, giữ nguyên `Event ID`.
   - Nếu không thể xác minh: Chuyển sự kiện sang trạng thái chờ nhân viên kiểm tra (`REVIEW_REQUIRED`), không tự động kết luận đồ đã bị lấy đi.
8. **Tuyên bố:** Hệ thống không khẳng định đã xác định chính xác danh tính chủ sở hữu của món đồ.

---

## 🏗️ 2. Kiến trúc Xử lý Hệ thống (Phân tách rõ ràng)

Hệ thống phân tách rành mạch các khái niệm thành các module chuyên biệt:

```
[ Camera Frame / Video ]
           │
           ▼
┌─────────────────────────┐
│ 1. Object Detection     │ ➔ YOLOv8 (phát hiện person & item_classes)
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 2. Object Tracking      │ ➔ ByteTrack (gán track_id tạm thời theo frame)
└──────────┬──────────────┘
           │
           ├──────────────────────────────┐
           ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│ 3. Seat Occupancy       │    │ 4. Occlusion Detection  │
│ (src/seats/seat_manager)│    │ (src/events/occlusion)  │
│ • ROI Polygon           │    │ • Case A: Ghế có người  │
│ • Overlap ratio         │    │ • Case B: Ghế không che │
│ • Temporal smoothing    │    │ • Case C: Người ngồi/dậy│
│ • Occupied / Vacant     │    │ • Case D: Mất detection │
└──────────┬──────────────┘    └──────────┬──────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
           ┌─────────────────────────────┐
           │ 5. Forgotten-Item Detector  │
           │ (src/events/forgotten_item) │
           │ • Phân biệt 3 loại thời gian│
           │ • Re-association bằng chứng │
           └──────────────┬──────────────┘
                          ▼
           ┌─────────────────────────────┐
           │ 6. Event Lifecycle Manager  │
           │ (src/events/event_manager)  │
           │ • State Machine (8 states)  │
           │ • Event ID ổn định          │
           │ • Snapshot & State History  │
           └──────────────┬──────────────┘
                          ▼
           ┌─────────────────────────────┐
           │ 7. Storage & Presentation   │
           │ (SQLite & Streamlit UI)     │
           └─────────────────────────────┘
```

### Các module chính:
- `src/seats/seat_config.py`: Quản lý cấu hình vùng ghế (Seat ROI polygon), tính toán giao cắt hình học giữa Bounding Box và đa giác ghế.
- `src/seats/seat_manager.py`: Ước lượng trạng thái ghế (`OCCUPIED`, `VACANT`, `UNKNOWN`) với cơ chế làm mượt theo thời gian (`SEAT_OCCUPIED_CONFIRM_SECONDS`, `SEAT_VACANT_CONFIRM_SECONDS`, `SEAT_HOLD_SECONDS`) chống nhiễu detection 1-frame.
- `src/events/occlusion_manager.py`: Phân tích 4 trường hợp che khuất (Case A, B, C, D) và so sánh bằng chứng (vị trí, khoảng cách, loại vật thể, thời gian) để quyết định tái liên kết sự kiện.
- `src/events/forgotten_item_detector.py`: Bộ phân tích tổng hợp, điều phối State Machine.
- `src/events/event_manager.py`: Quản lý danh tính sự kiện (`Event ID` ổn định không đổi), ghi nhận `state_history`, lưu trữ ảnh bằng chứng snapshot.
- `src/video/video_writer.py`: Xuất video với chú thích trực quan: vẽ vùng Seat ROI, màu sắc phản ánh trạng thái ghế, hiển thị nhãn trạng thái và vị trí vật thể bị che khuất.
- `src/database/db.py` & `src/database/event_repository.py`: Lưu trữ SQLite an toàn với cơ chế tự động migration schema.
- `app.py`: Giao diện web Streamlit cao cấp, quản lý ROI, theo dõi thời gian thực và bộ lọc sự kiện.

---

## 🔄 3. State Machine Vòng đời Sự kiện

Mọi sự kiện đồ vật đều tuân theo State Machine nghiêm ngặt:

| Trạng thái | Mã định danh | Ý nghĩa |
|---|---|---|
| **OBSERVED** | `observed` | Vật thể được phát hiện trong khung hình khi hành khách vẫn đang ngồi tại ghế. |
| **ITEM_LEFT** | `item_left` | Ghế đã được xác nhận trống, vật thể vẫn còn lại trên ghế. Bắt đầu đếm thời gian. |
| **SUSPECTED** | `suspected` | Vật thể đứng yên trên ghế trống vượt quá ngưỡng tối thiểu (`min_stationary_time`). |
| **OCCLUDED** | `occluded` | Vật thể mất detection và có bằng chứng vùng quan sát/ghế bị người khác che khuất. |
| **RECHECK** | `recheck` | Ghế đã quan sát được trở lại, đang trong giai đoạn xác minh (`recheck_duration`). |
| **REVIEW_REQUIRED** | `review_required` | Ghế đã mở nhưng không thấy lại vật thể; cần nhân viên nhà xe kiểm tra thực tế. |
| **RESOLVED** | `resolved` / `confirmed` | Sự kiện kết thúc (nhân viên xác nhận đã xử lý đồ bỏ quên). |
| **DISMISSED** | `dismissed` | Nhân viên loại bỏ cảnh báo (ví dụ hành khách chỉ tạm đặt đồ hoặc vật cố định). |

### Quy tắc chuyển đổi:
1. **Không cho phép** một detection bị mất tạm thời làm chuyển sự kiện thẳng sang `RESOLVED`.
2. **Không tự chuyển** `OCCLUDED` sang `RESOLVED` chỉ vì hết thời gian che khuất.
3. **Phân biệt rõ 3 loại thời gian:**
   - `observed_duration`: Thời gian thực sự nhìn thấy vật thể.
   - `occluded_duration`: Thời gian vùng ghế bị che khuất.
   - `vacant_duration`: Thời gian ghế trống mà vật thể vẫn tồn tại.
   *(Không được cộng gộp thời gian che khuất như thể vẫn nhìn thấy đồ vật).*

---

## 🪑 4. Quản lý Sơ đồ Ghế Xe Buýt (Seat ROI) & Ký hiệu Số Ghế Trực tiếp

Xe buýt có nhiều cấu hình ghế khác nhau (xe 16 chỗ, 29 chỗ, 45 chỗ...). Hệ thống hỗ trợ:
1. **Bộ sinh nhanh sơ đồ ghế xe buýt:** Tự động tạo 4, 8, 12, 16, 24... ghế tùy chọn theo số hàng và số ghế mỗi bên (1-1, 2-1, 2-2).
2. **Ký hiệu số ghế trực quan:** Cho phép chọn kiểu đặt tên (`Ghế 1A, 1B, 2A...` hoặc `SEAT_01, SEAT_02...`).
3. **Hiển thị trực tiếp lên Video & Camera:** Mọi vật thể hoặc người khi phát hiện sẽ được **gắn mã số ghế trực tiếp trên Bounding Box**, ví dụ:
   - `[Ghế 1A] Ba lô #2 85% ⚠ NGHI NGỜ`
   - `[Ghế 2B] Ví #5 88% 🛡 BỊ CHE KHUẤT`
   - `[Ghế 1A] Người #10 92%`
4. **Tọa độ chuẩn hóa `[0.0, 1.0]`:** Tự động co dãn theo mọi kích thước camera (720p, 1080p, 4K).
5. **Tránh gán trùng:** Thuật toán ưu tiên kiểm tra tâm vật thể nằm trong ghế, sau đó mới tính tỷ lệ giao cắt lớn nhất (`overlap_ratio`).

---

## 🎥 5. Phân tích Trực tiếp qua Live Webcam & Video File

Hệ thống hỗ trợ 2 chế độ phân tích:
- **📁 File Video:** Tải lên video đã quay từ camera trần xe buýt (`.mp4`, `.mov`, `.avi`), phân tích và xuất video có vẽ vùng ghế và chú thích.
- **🎥 Live Webcam (Thời gian thực):** Mở camera máy tính hoặc camera ngoài gắn qua USB/cổng mạng, theo dõi trực tiếp các chuyển động, trạng thái ghế có người/trống, và cảnh báo đồ bỏ quên/che khuất theo thời gian thực.

---

## 🚀 6. Hướng dẫn Cài đặt & Chạy Hệ thống

### Bước 1: Kích hoạt môi trường ảo
```bash
cd /Users/thienduc/Desktop/Code/findlostitems
source venv/bin/activate
```

### Bước 2: Chạy toàn bộ Unit Tests
```bash
pytest -v
```
*(Hiện có 46/46 test cases bao phủ toàn bộ 10 kịch bản nghiệp vụ bắt buộc, kiểm tra hình học ROI, bộ sinh lưới ghế, database CRUD và pipeline video).*

### Bước 3: Khởi động giao diện Streamlit
```bash
./venv/bin/streamlit run app.py
```
Mở trình duyệt tại: `http://localhost:8501`.


---

## 🧪 6. Kết quả Kiểm thử Nghiệp vụ (10 Kịch bản Bắt buộc)

Toàn bộ 10 kịch bản đều được kiểm thử tự động trong `tests/test_occlusion_and_state_machine.py`:

| STT | Tình huống kiểm thử | Kết quả mong đợi | Trạng thái Test |
|---|---|---|:---:|
| 1 | Đồ vật trên ghế có người ngồi | Ở trạng thái `OBSERVED`, chưa tạo cảnh báo | ✅ PASSED |
| 2 | Người đứng lên, đồ vật còn trên ghế trống | Ghế chuyển `VACANT`, sự kiện sang `ITEM_LEFT` | ✅ PASSED |
| 3 | Đồ vật đứng yên trên ghế trống đủ thời gian | Đủ điều kiện tạo cảnh báo `SUSPECTED` | ✅ PASSED |
| 4 | Người mới ngồi xuống che khuất đồ vật | Chuyển `OCCLUDED`, **giữ nguyên Event ID** | ✅ PASSED |
| 5 | Người ngồi che đứng lên, đồ xuất hiện lại | Khôi phục sự kiện cũ bằng evidence matching | ✅ PASSED |
| 6 | Người ngồi che đứng lên, đồ không thấy lại | Chuyển `REVIEW_REQUIRED`, không tự kết luận mất | ✅ PASSED |
| 7 | Detection bị mất 1 frame do nhiễu | Temporal smoothing giữ trạng thái, không tạo mới | ✅ PASSED |
| 8 | Vùng ghế không che nhưng đồ biến mất | Vào `RECHECK` rồi chuyển `REVIEW_REQUIRED` | ✅ PASSED |
| 9 | Mất ByteTrack ID rồi xuất hiện lại với ID khác | Khớp bằng chứng vị trí/lớp, không nhân bản event | ✅ PASSED |
| 10 | Nhiều ghế có sự kiện đồng thời | Mỗi ghế có vòng đời sự kiện hoàn toàn độc lập | ✅ PASSED |

---

## 📊 7. Báo cáo Hiện trạng Chức năng & Giới hạn Thực tế

### ✅ Những chức năng đã chạy thật & hoạt động hoàn chỉnh:
1. **Pipeline Object Detection & ByteTrack:** Phát hiện người và các lớp đồ dùng (balo, túi xách, vali, điện thoại, ô dù...).
2. **Seat ROI Manager:** Xác thực polygon, tính toán overlap ratio theo sub-mask tối ưu tốc độ, chuyển đổi pixel/normalized, làm mượt trạng thái ghế.
3. **Occlusion-Aware Engine:** Đánh giá 4 trường hợp che khuất và cơ chế so sánh bằng chứng vị trí hình học.
4. **State Machine 8 trạng thái:** Đảm bảo tính toàn vẹn vòng đời sự kiện và nhật ký `state_history`.
5. **SQLite Schema & Migration:** Tự động phát hiện và thêm các cột mới mà không mất dữ liệu cũ.
6. **Video Writer Chú thích Đầy đủ:** Vẽ vùng ghế, màu sắc ghế và các bounding box (kể cả vị trí vật thể bị che khuất).
7. **Streamlit UI Đa năng:** Xem dashboard, phân tích video kèm thanh tiến trình, lọc sự kiện, chỉnh sửa JSON vùng ghế trực tiếp.

### ⚠️ Giới hạn Thực tế Cần Lưu ý:
1. **Không định danh chủ sở hữu:** Hệ thống không dùng nhận diện khuôn mặt và không liên kết danh tính người với món đồ; chỉ ghi nhận tương tác vị trí giữa người và vật thể.
2. **Điểm mù & Che khuất hoàn toàn:** Nếu một người lấy đồ và giấu vào túi áo/ba lô ngay trong lúc cơ thể che khuất hoàn toàn góc nhìn của camera, AI không thể quan sát được hành động lấy đồ đó. Khi người đó rời đi, hệ thống chỉ biết đồ vật không còn và sẽ đưa vào diện `REVIEW_REQUIRED` để bảo vệ tài sản hành khách.
3. **Hiệu chuẩn tham số:** Các ngưỡng thời gian (`min_stationary_time`, `hold_sec`, `overlap_threshold`) là tham số mặc định được thiết kế để hạn chế cảnh báo sai; người vận hành cần căn chỉnh cho phù hợp với tốc độ dừng/đón khách của từng tuyến xe buýt thực tế.
