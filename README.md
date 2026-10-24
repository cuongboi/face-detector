# Face Attendance (local, Apple Silicon)

Điểm danh bằng khuôn mặt, chạy hoàn toàn offline trên MacBook M1:
**Apple Vision** (detect + 76 landmark + head pose, quét lưới tile cho mặt xa 5–10m) →
align 5 điểm chuẩn ArcFace → **ArcFace w600k_r50 (Core ML, Neural Engine)** → **Qdrant** → **Gradio**.
Không cần PyTorch / ONNX Runtime lúc chạy; chỉ cần macOS 13+ trên Apple Silicon.

## Luồng kỹ thuật

Mỗi khung hình (webcam trình duyệt hoặc RTSP) đi qua `FacePipeline.analyze`
([core/pipeline.py](core/pipeline.py)), dùng chung cho Đăng ký và Điểm danh:

```mermaid
flowchart TD
    SRC["Khung hình BGR<br/>webcam trình duyệt / RTSP"] --> BR["vision_bridge<br/>numpy → CGImage"]
    BR --> DET["Apple Vision · Neural Engine<br/>FaceRectangles rev3: toàn khung + lưới tile NxN"]
    DET --> NMS["Lọc vision_conf, min_face_size<br/>NMS gộp mặt trùng giữa các tile"]
    NMS --> LM["Vision FaceLandmarks<br/>76 điểm + yaw/pitch"]
    LM --> KPS["kps5: 2 mắt, mũi, 2 khoé miệng"]
    KPS --> ENH{"Crop mặt tối / mờ?"}
    ENH -- "tối" --> DN["NAFNet-SIDD + CLAHE<br/>Core ML"]
    ENH -- "mờ" --> DB["NAFNet-GoPro<br/>Core ML"]
    ENH -- "bình thường" --> AL
    DN --> AL["align_5pt<br/>warp 112×112 chuẩn ArcFace"]
    DB --> AL
    AL --> EMB["ArcFace w600k_r50<br/>Core ML · Neural Engine, batch ≤ 16"]
    EMB --> VEC["Vector 512-d, chuẩn hoá L2"]
    VEC --> MODE{"Tab"}
    MODE -- "Đăng ký" --> ENR["EnrollmentSession<br/>phân loại góc theo yaw/pitch"]
    ENR --> QA[("Qdrant<br/>1 vector / góc / người")]
    ENR --> SA[("SQLite<br/>người")]
    MODE -- "Điểm danh" --> SR["Qdrant search top-k<br/>cosine ≥ match_threshold"]
    SR --> TR["AttendanceService<br/>track IoU + vote_frames"]
    TR --> SB[("SQLite<br/>vào/ra + ảnh JPEG base64")]
    TR --> UI["Gradio: khung + nhãn trạng thái"]
```

Quyết định cho mỗi mặt, mỗi khung hình (`AttendanceService._decide`, [core/attendance.py](core/attendance.py)):

```mermaid
flowchart TD
    M{"Khớp ai?<br/>cosine ≥ match_threshold"} -- "không" --> U["unknown"]
    M -- "có" --> W{"Đang chờ<br/>next_write?"}
    W -- "có" --> C["cooldown"]
    W -- "không" --> V{"vote_frames khung liên tiếp<br/>cùng một người?"}
    V -- "chưa" --> VF["verifying"]
    V -- "đủ" --> R["record_sighting<br/>(điểm TB các vote)"]
    R -- "lần đầu trong ngày /<br/>điểm cao hơn trong in_window" --> IN["checked_in<br/>chờ sighting_interval"]
    R -- "sau in_window" --> OUT["checked_out<br/>ghi đè Ra, chờ cooldown"]
    R -- "không ghi gì" --> C
```

## Luồng sử dụng

```mermaid
flowchart LR
    A["Cài đặt<br/>uv sync · download_models · Qdrant"] --> B["uv run python app.py"]
    B --> C["Tab Đăng ký<br/>họ tên + mã NV"]
    C --> D["Theo hướng dẫn 7 góc<br/>front → left → full_left → right → full_right → up → down"]
    D --> E{"Đủ 7 góc?"}
    E -- "chưa" --> D
    E -- "đủ" --> F["💾 Lưu"]
    F --> G["Tab Điểm danh<br/>webcam hoặc Camera IP (RTSP)"]
    G --> H["Đứng trước camera<br/>khớp liên tiếp vote_frames khung"]
    H --> I["Ghi Vào (lần đầu trong ngày)<br/>Ra (các lần sau, ghi đè)"]
    I --> J["Tab Quản lý & Lịch sử<br/>xem ảnh vào/ra · xoá người · export CSV"]
```

## Cài đặt

```bash
uv sync                                   # Python 3.11 + deps
uv run --group convert python scripts/download_models.py  # tải ~360 MB + chuyển sang Core ML
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
- **Điểm danh** (tab mặc định): một người được xác nhận khi khớp liên tiếp `FA_VOTE_FRAMES`
  khung hình. Mỗi người một dòng mỗi ngày (vào lần đầu / ra lần cuối):
  - Lần khớp đầu tiên trong ngày → ghi **Vào** + ảnh.
  - Trong `FA_IN_WINDOW_SECONDS` (60s) sau khi vào, nếu khớp với điểm cao hơn → thay ảnh/điểm
    vào, **giữ nguyên giờ vào**.
  - Khớp sau 60s → ghi **Ra** + ảnh; mọi lần khớp sau đó (điểm bất kỳ) ghi đè **Ra**
    (giờ, điểm, ảnh), tối đa mỗi `FA_COOLDOWN_SECONDS`.
  Ảnh là khung hình gốc (không lật, không khung), JPEG rộng `FA_SNAPSHOT_WIDTH`=960px, base64
  trong SQLite. Bấm một dòng để xem ảnh vào; bấm ô **Ra** để xem ảnh ra.
- **Quản lý & Lịch sử**: xoá người, xem vào/ra theo ngày kèm ảnh, export CSV
  (bảng vào/ra + nhật ký các lần ghi, không kèm ảnh).

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
| `FA_VISION_TILE_GRID` | 3        | Lưới NxN quét thêm (Vision chỉ thấy mặt ≥ ~5% chiều rộng ảnh). 3 → mặt ~40px ở 1080p, ~60 ms; 1 → chỉ toàn khung, ~10 ms, mặt ≥ ~100px; 4 nếu vẫn miss |
| `FA_VISION_CONF`      | 0.3      | Hạ nếu miss, tăng nếu nhận nhầm vật thể là mặt |
| `FA_MIN_FACE_SIZE`    | 40       | px, bỏ mặt quá nhỏ để nhận diện tin cậy |
| `FA_MATCH_THRESHOLD`  | 0.5      | Cosine; tăng nếu nhận nhầm, giảm nếu không nhận ra |
| `FA_VOTE_FRAMES`      | 3        | Chống false positive |

Toàn bộ tham số: [utils/config.py](utils/config.py).

## Thiếu sáng / camera mờ (NAFNet)

Trước khi căn chỉnh + embed, vùng mặt được phục hồi bằng [NAFNet](https://github.com/megvii-research/NAFNet)
(width32, Core ML FP16 trên Neural Engine, ~15 ms/mặt). Chỉ chạy khi cần:

- Mặt tối → NAFNet-SIDD khử nhiễu, sau đó CLAHE tăng sáng.
- Mặt mờ → NAFNet-GoPro khử mờ.

Mặt được xử lý có nhãn `+denoise` / `+deblur` trong tab Điểm danh.

| Biến `.env`                 | Mặc định | Ghi chú |
|-----------------------------|----------|---------|
| `FA_ENHANCE_ENABLED`        | true     | `false` để tắt (không cần model NAFNet) |
| `FA_ENHANCE_DARK_LUMA`      | 70       | Độ sáng TB (0–255) dưới mức này → khử nhiễu |
| `FA_ENHANCE_BLUR_SHARPNESS` | 60       | Phương sai Laplacian dưới mức này → khử mờ |
| `FA_ENHANCE_MAX_SIDE`       | 192      | Crop lớn hơn bị thu nhỏ trước để giữ tốc độ |

## Phát triển

```bash
uv run pytest        # test logic chạy không cần Docker; test detector cần macOS
uv run ruff check .
```

Cấu trúc: `core/` (vision_detector, vision_bridge, aligner, head_pose, embedder, vector_store, repository,
attendance, enrollment, pipeline), `utils/` (config, logger, face_utils), `app.py` (UI),
`scripts/convert_models.py` (ONNX/PyTorch → Core ML, kiểm tra sai số so với model gốc).
SQLite (`data/attendance.db`) là nguồn dữ liệu chính cho người/lịch sử; Qdrant chỉ lưu vector
(7 vector/người, mỗi góc một vector).
