# Face Attendance (local, Apple Silicon)

Điểm danh bằng khuôn mặt, chạy hoàn toàn offline trên MacBook M1:
**YOLOv11n-face** (detect xa 5–10m) → **MediaPipe FaceLandmarker** (landmark + head pose) →
align 5 điểm chuẩn ArcFace → **ArcFace w600k_r50 (ONNX)** → **Qdrant** → **Gradio**.
Chi tiết thiết kế: [plan.md](plan.md).

## Cài đặt

```bash
uv sync                                   # Python 3.11 + deps
uv run python scripts/download_models.py  # ~185 MB vào data/models/
docker compose up -d                      # Qdrant :6333
cp .env.example .env                      # tuỳ chọn: chỉnh tham số
uv run python app.py                      # http://127.0.0.1:7860
```

## Sử dụng

- **Đăng ký**: nhập họ tên + mã NV, nhìn webcam và làm theo hướng dẫn qua 5 góc
  `front → left → right → up → down`. Ảnh tự chụp khi giữ đúng góc ~0.5s (hoặc bấm 📸).
  Mỗi ảnh được kiểm tra kích thước, độ nét và phải khớp với ảnh chính diện. Đủ 5 góc → 💾 Lưu.
- **Điểm danh**: một người được ghi nhận khi khớp liên tiếp `FA_VOTE_FRAMES` khung hình,
  sau đó cooldown `FA_COOLDOWN_SECONDS`.
- **Quản lý & Lịch sử**: xoá người, xem lịch sử theo ngày, export CSV.

## Tune cho khoảng cách 5–10m

| Biến `.env`           | Mặc định | Ghi chú |
|-----------------------|----------|---------|
| `FA_YOLO_IMGSZ`       | 960      | Tăng nếu miss mặt ở xa (chậm hơn) |
| `FA_YOLO_CONF`        | 0.4      | Hạ xuống 0.3 nếu miss |
| `FA_MIN_FACE_SIZE`    | 40       | px, bỏ mặt quá nhỏ để nhận diện tin cậy |
| `FA_MATCH_THRESHOLD`  | 0.5      | Cosine; tăng nếu nhận nhầm, giảm nếu không nhận ra |
| `FA_VOTE_FRAMES`      | 3        | Chống false positive |

Toàn bộ tham số: [utils/config.py](utils/config.py).

## Phát triển

```bash
uv run pytest        # test logic chạy không cần Docker; test detector cần model
uv run ruff check .
```

Cấu trúc: `core/` (detector, aligner, head_pose, embedder, vector_store, repository,
attendance, enrollment, pipeline), `utils/` (config, logger, face_utils), `app.py` (UI).
SQLite (`data/attendance.db`) là nguồn dữ liệu chính cho người/lịch sử; Qdrant chỉ lưu vector
(5 vector/người, mỗi góc một vector).
