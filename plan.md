# Plan: Hệ thống Điểm danh Gương mặt (Local trên MacBook Air M1)

## 1. Mục tiêu

Xây dựng hệ thống điểm danh bằng nhận diện khuôn mặt chạy **local** trên MacBook Air M1 với các yêu cầu:

- Nhận diện từ khoảng cách **5–10 mét**
- Sử dụng **YOLO + MediaPipe hybrid**
- Giao diện **Gradio**
- Có thể **đăng ký khuôn mặt mới**
- Lưu vector bằng **Qdrant** (chạy qua Docker)
- Hoạt động hoàn toàn offline sau khi cài đặt

---

## 2. Kiến trúc tổng quan

```
Camera / Webcam
      ↓
YOLO Face Detection (detect xa 5–10m)
      ↓
Crop + Expand bounding box
      ↓
MediaPipe Face Mesh (landmark 3D + refine)
      ↓
Align Face → Embedding Model
      ↓
Qdrant (search / insert vector)
      ↓
Gradio UI (Đăng ký + Điểm danh + Lịch sử)
```

### Luồng chính

**Đăng ký khuôn mặt mới:**
```
Ảnh / Webcam → YOLO detect → MediaPipe refine + landmark
→ Align → Embedding → Insert Qdrant (kèm payload: tên, mã NV, thời gian)
```

**Điểm danh real-time:**
```
Frame → YOLO → MediaPipe → Align → Embedding
→ Search Qdrant (top-1, cosine)
→ Nếu score > threshold → Ghi nhận điểm danh + cooldown
```

---

## 3. Công nghệ sử dụng

| Thành phần              | Công nghệ                          | Lý do chọn |
|-------------------------|------------------------------------|----------|
| Face Detection (xa)     | YOLOv8n-face / YOLOv11n-face      | Detect tốt ở khoảng cách xa |
| Landmark 3D + Refine    | MediaPipe Face Mesh                | 468 điểm 3D, align chính xác |
| Embedding               | InsightFace (buffalo_l) ONNX       | Chất lượng cao, chạy ổn trên M1 |
| Vector Database         | Qdrant (Docker)                    | Nhanh, cosine similarity, dễ dùng |
| UI                      | Gradio                             | Nhanh, đẹp, hỗ trợ webcam streaming |
| Runtime                 | Python 3.11 + onnxruntime          | Tối ưu Apple Silicon |
| Lưu lịch sử             | SQLite (hoặc payload Qdrant)       | Đơn giản, đủ dùng |

---

## 4. Cấu trúc thư mục dự án

```
face_attendance/
├── docker-compose.yml              # Qdrant
├── requirements.txt
├── plan.md                         # File này
├── README.md
├── app.py                          # Gradio main application
├── core/
│   ├── hybrid_detector.py          # YOLO + MediaPipe hybrid
│   ├── embedder.py                 # InsightFace / ONNX embedding
│   ├── aligner.py                  # Align face từ landmark MediaPipe
│   ├── qdrant_client.py            # Wrapper Qdrant
│   └── attendance.py               # Logic điểm danh + cooldown
├── utils/
│   ├── face_utils.py               # Crop, normalize, draw
│   ├── config.py                   # Cấu hình chung
│   └── logger.py
├── data/
│   ├── attendance.db               # SQLite lịch sử điểm danh
│   └── models/                     # YOLO + embedding models
└── tests/
    └── test_detector.py
```

---

## 5. Chi tiết các module chính

### 5.1. Hybrid Detector (`core/hybrid_detector.py`)

**Luồng:**
1. YOLO detect face trên frame gốc (imgsz 640–960)
2. Mở rộng bounding box (expand_ratio ≈ 0.25–0.35)
3. Crop vùng face
4. Chạy MediaPipe Face Mesh trên crop để lấy 468 landmark 3D
5. Trả về: `bbox`, `confidence`, `landmarks_3d`, `crop`

**Tham số quan trọng cho khoảng cách 5–10m:**

| Tham số              | Giá trị khuyến nghị     | Ghi chú |
|----------------------|-------------------------|--------|
| YOLO `imgsz`         | 640 → 800 / 960         | Tăng để detect xa hơn |
| `conf_threshold`     | 0.35 – 0.45             | Hạ nếu miss nhiều |
| `expand_ratio`       | 0.25 – 0.35             | Đủ không gian cho MediaPipe |
| `min_face_size`      | 40–60 px                | Bỏ face quá nhỏ |
| Webcam resolution    | ≥ 1280×720              | Rất quan trọng |

### 5.2. Face Aligner (`core/aligner.py`)

- Sử dụng 5 điểm chính từ MediaPipe (2 mắt, mũi, 2 khóe miệng) hoặc full 468 điểm
- Affine transform → chuẩn hóa về 112×112 hoặc 160×160
- Giúp embedding ổn định hơn rất nhiều so với crop thẳng

### 5.3. Embedder (`core/embedder.py`)

- Model khuyến nghị: **InsightFace buffalo_l** (512-dim)
- Chạy bằng `onnxruntime` (tối ưu M1)
- Input: ảnh đã align 112×112
- Output: vector L2-normalized

### 5.4. Qdrant Client (`core/qdrant_client.py`)

- Collection: `faces`
- Vector size: 512
- Distance: Cosine
- Payload: `name`, `employee_id`, `created_at`, `image_path` (tùy chọn)

**Các thao tác:**
- `insert_face(embedding, payload)`
- `search(embedding, threshold=0.5, top_k=1)`
- `list_all()`, `delete_by_id()`

### 5.5. Attendance Logic (`core/attendance.py`)

- Threshold cosine: **0.45 – 0.55** (cần tune thực tế)
- Cooldown: 30–60 giây / người để tránh spam
- Ghi log vào SQLite: `id, name, employee_id, timestamp, score, camera_id`

---

## 6. Giao diện Gradio (`app.py`)

### 3 Tab chính:

**Tab 1: Đăng ký khuôn mặt mới**
- Webcam hoặc Upload ảnh
- Input: Họ tên + Mã nhân viên
- Preview face đã detect + align
- Nút "Đăng ký" → lưu vào Qdrant
- Thông báo thành công / thất bại

**Tab 2: Điểm danh Real-time**
- Webcam streaming
- Hiển thị bounding box + tên + score
- Log điểm danh gần nhất
- Có thể bật/tắt âm thanh thông báo

**Tab 3: Quản lý & Lịch sử**
- Danh sách người đã đăng ký (có nút xóa)
- Lịch sử điểm danh theo ngày
- Export CSV / Excel (tùy chọn)

---

## 7. Cài đặt & Chạy trên Mac Air M1

### 7.1. Chuẩn bị môi trường

```bash
# Tạo virtualenv
python3.11 -m venv venv
source venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

**requirements.txt (dự kiến):**
```
ultralytics>=8.3.0
mediapipe>=0.10.14
opencv-python-headless
gradio>=4.44.0
qdrant-client
insightface
onnxruntime          # hoặc onnxruntime-silicon nếu cần
numpy
Pillow
sqlite3              # built-in
```

### 7.2. Chạy Qdrant

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    restart: unless-stopped
```

```bash
docker compose up -d
```

### 7.3. Tải model

- YOLO face: `yolov8n-face.pt` (từ repo uy tín như akanametov/yolo-face hoặc derronqi/yolov8-face)
- InsightFace buffalo_l: tự động tải khi dùng `insightface`

### 7.4. Chạy ứng dụng

```bash
python app.py
```

Gradio sẽ mở tại `http://127.0.0.1:7860`

---

## 8. Tối ưu hiệu năng trên M1

| Kỹ thuật                        | Mô tả                                      | Mức độ ưu tiên |
|--------------------------------|--------------------------------------------|----------------|
| ONNX Runtime                   | Chạy embedding & YOLO bằng ONNX            | Cao            |
| Giảm tần suất detect           | Chỉ chạy YOLO mỗi 2–3 frame                | Cao            |
| Resolution hợp lý              | Webcam 1280×720, YOLO imgsz=640/800        | Cao            |
| Multi-thread / async           | Tách capture và inference                  | Trung bình     |
| Quantization (nếu cần)         | INT8 cho embedding model                   | Thấp           |

---

## 9. Các rủi ro & cách xử lý

| Rủi ro                          | Giải pháp |
|--------------------------------|---------|
| Detect yếu ở 8–10m              | Tăng `imgsz`, hạ `conf`, dùng camera zoom / đặt cao hơn |
| Ánh sáng kém / ngược sáng       | Thêm CLAHE hoặc normalize trước embedding |
| False positive                  | Tăng threshold + yêu cầu 2–3 frame liên tiếp |
| Nhiều người cùng lúc            | Xử lý multi-face, điểm danh tất cả hoặc người gần nhất |
| Model YOLO face không có sẵn    | Dùng YOLOv8n + fine-tune hoặc tải từ repo face chuyên dụng |
| MediaPipe miss trên crop nhỏ    | Tăng `expand_ratio` và đảm bảo crop đủ lớn |

---

## 10. Roadmap phát triển

### Phase 1 – MVP (1–2 ngày)
- [x] Hybrid Detector (YOLO + MediaPipe)
- [ ] Embedder + Aligner
- [ ] Qdrant client cơ bản
- [ ] Gradio: Đăng ký + Điểm danh real-time
- [ ] Lưu lịch sử đơn giản (SQLite)

### Phase 2 – Ổn định (2–3 ngày)
- [ ] Cooldown + anti-spam
- [ ] Multi-face handling
- [ ] Tune threshold thực tế
- [ ] Tab Quản lý + Lịch sử
- [ ] Export báo cáo

### Phase 3 – Nâng cao (tùy chọn)
- [ ] Liveness detection nhẹ
- [ ] Hỗ trợ camera IP / RTSP
- [ ] Dashboard thống kê
- [ ] Âm thanh thông báo điểm danh
- [ ] Chạy như service (launchd trên macOS)

---

## 11. Ghi chú quan trọng

- **Khoảng cách 5–10m** phụ thuộc rất nhiều vào chất lượng camera và ánh sáng. Webcam laptop thông thường chỉ ổn ~5–7m. Nên test thực tế và điều chỉnh `imgsz` + resolution.
- Nên thu thập 3–5 ảnh / người khi đăng ký (góc khác nhau) để tăng độ chính xác.
- Threshold cosine cần được tune trên dữ liệu thật của môi trường triển khai.
- Toàn bộ hệ thống chạy local, không gửi dữ liệu ra ngoài.

---

**Ngày tạo plan:** 24/10/2026  
**Môi trường mục tiêu:** MacBook Air M1 (Apple Silicon)  
**Trạng thái:** Sẵn sàng bắt đầu code Phase 1
