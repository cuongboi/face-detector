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
uv run python scripts/make_cert.py        # tuỳ chọn: HTTPS để dùng camera điện thoại
uv run python app.py                      # https://localhost:7860 (http nếu không có cert)
```

Điện thoại chỉ cho dùng camera qua HTTPS. Sau khi tạo cert, mở `https://<IP-máy-Mac>:7860`
trên điện thoại (cùng Wi-Fi), chấp nhận cảnh báo cert, dùng tab **Đăng ký**; tab **Điểm danh**
vẫn chạy trên Mac. Đổi mạng/IP → chạy lại `make_cert.py`.

## Sử dụng

- **Đăng ký**: nhập họ tên + mã NV, nhìn webcam và làm theo hướng dẫn qua 7 góc
  `front → left → full_left → right → full_right → up → down`
  (`left/right` quay ~20–50°, `full_left/full_right` quay hẳn ~50–85°, tuỳ chỉnh bằng
  `FA_FULL_YAW_MIN` / `FA_FULL_YAW_MAX`). Ảnh tự chụp khi giữ đúng góc ~0.5s (hoặc bấm 📸).
  Mỗi ảnh được kiểm tra kích thước, độ nét và phải khớp với một góc đã chụp. Đủ 7 góc → 💾 Lưu.
- **Điểm danh**: một người được ghi nhận khi khớp liên tiếp `FA_VOTE_FRAMES` khung hình,
  sau đó cooldown `FA_COOLDOWN_SECONDS`.
- **Quản lý & Lịch sử**: xoá người, xem lịch sử theo ngày, export CSV.

## Camera IP (RTSP)

Tab **Điểm danh** → chọn **Camera IP (RTSP)**, nhập URL rồi bấm **▶ Kết nối** (hoặc Enter):

```
rtsp://user:pass@192.168.1.10:554/stream1
```

Đường dẫn tuỳ hãng (Hikvision `/Streaming/Channels/101`, Dahua `/cam/realmonitor?channel=1&subtype=0`, …).
Server (máy Mac) tự đọc luồng nên camera phải truy cập được từ máy chạy app, không phải từ điện thoại.
Đặt sẵn URL bằng `FA_RTSP_URL`. Luồng đi qua TCP, tự kết nối lại khi rớt mạng; mỗi lúc chỉ mở
một camera IP (kết nối camera mới sẽ đóng camera cũ). Mật khẩu được che trong log và giao diện.

## Tune cho khoảng cách 5–10m

| Biến `.env`           | Mặc định | Ghi chú |
|-----------------------|----------|---------|
| `FA_YOLO_IMGSZ`       | 960      | Tăng nếu miss mặt ở xa (chậm hơn) |
| `FA_YOLO_CONF`        | 0.4      | Hạ xuống 0.3 nếu miss |
| `FA_MIN_FACE_SIZE`    | 40       | px, bỏ mặt quá nhỏ để nhận diện tin cậy |
| `FA_MATCH_THRESHOLD`  | 0.5      | Cosine; tăng nếu nhận nhầm, giảm nếu không nhận ra |
| `FA_VOTE_FRAMES`      | 3        | Chống false positive |

Toàn bộ tham số: [utils/config.py](utils/config.py).

## Thiếu sáng / camera mờ (NAFNet)

Trước khi căn chỉnh + embed, vùng mặt được phục hồi bằng [NAFNet](https://github.com/megvii-research/NAFNet)
(width32, chạy trên GPU M1 qua PyTorch MPS, ~40–60 ms/mặt). Chỉ chạy khi cần:

- Mặt tối → NAFNet-SIDD khử nhiễu, sau đó CLAHE tăng sáng.
- Mặt mờ → NAFNet-GoPro khử mờ.

Mặt được xử lý có nhãn `+denoise` / `+deblur` trong tab Điểm danh.

| Biến `.env`                 | Mặc định | Ghi chú |
|-----------------------------|----------|---------|
| `FA_ENHANCE_ENABLED`        | true     | `false` để tắt (không cần tải weights NAFNet) |
| `FA_ENHANCE_DARK_LUMA`      | 70       | Độ sáng TB (0–255) dưới mức này → khử nhiễu |
| `FA_ENHANCE_BLUR_SHARPNESS` | 60       | Phương sai Laplacian dưới mức này → khử mờ |
| `FA_ENHANCE_MAX_SIDE`       | 192      | Crop lớn hơn bị thu nhỏ trước để giữ tốc độ |

## Phát triển

```bash
uv run pytest        # test logic chạy không cần Docker; test detector cần model
uv run ruff check .
```

Cấu trúc: `core/` (detector, aligner, head_pose, embedder, vector_store, repository,
attendance, enrollment, pipeline), `utils/` (config, logger, face_utils), `app.py` (UI).
SQLite (`data/attendance.db`) là nguồn dữ liệu chính cho người/lịch sử; Qdrant chỉ lưu vector
(5 vector/người, mỗi góc một vector).
