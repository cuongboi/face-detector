"""YOLO-face for far-range detection, MediaPipe FaceLandmarker for landmark refinement + pose."""

import threading

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from core.aligner import kps5_from_mediapipe
from core.head_pose import pose_from_kps5, pose_from_matrix
from core.types import FaceDetection
from utils.config import Settings
from utils.face_utils import apply_clahe, expand_bbox
from utils.logger import get_logger

log = get_logger(__name__)

# MediaPipe's internal detector wants the face to fill a reasonable part of the input.
MP_INPUT_SIZE = 256


class HybridDetector:
    def __init__(self, settings: Settings):
        self.s = settings
        self.yolo = YOLO(str(settings.model_path(settings.yolo_model)))
        self.device = settings.yolo_device
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(settings.model_path(settings.landmarker_model))
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        # Neither YOLO nor MediaPipe are safe to call concurrently (Gradio uses worker threads).
        self._lock = threading.Lock()

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        with self._lock:
            return self._detect(frame_bgr)

    def _detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        h, w = frame_bgr.shape[:2]
        result = self.yolo.predict(
            frame_bgr, imgsz=self.s.yolo_imgsz, conf=self.s.yolo_conf,
            device=self.device, verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        # Face models trained as pose models expose 5 keypoints; plain detectors don't.
        kps_all = None
        if getattr(result, "keypoints", None) is not None and result.keypoints.xy.shape[1] >= 5:
            kps_all = result.keypoints.xy.cpu().numpy()[:, :5]

        faces = []
        for i, (box, score) in enumerate(zip(boxes, scores, strict=True)):
            det = FaceDetection(bbox=box, score=float(score))
            if det.size < self.s.min_face_size:
                continue
            yolo_kps = kps_all[i] if kps_all is not None else None
            crop_box = expand_bbox(box, self.s.expand_ratio, w, h)
            if not self._refine_with_mediapipe(frame_bgr, crop_box, det) and yolo_kps is not None:
                det.kps5 = yolo_kps.astype(np.float32)
                det.yaw, det.pitch = pose_from_kps5(det.kps5)
                det.source = "yolo"
            if det.kps5 is not None:
                faces.append(det)
        return faces

    def _refine_with_mediapipe(
        self, frame_bgr: np.ndarray, crop_box: tuple[int, int, int, int], det: FaceDetection
    ) -> bool:
        x1, y1, x2, y2 = crop_box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        if self.s.use_clahe:
            crop = apply_clahe(crop)
        # Small far-away crops are upscaled; MediaPipe fails on tiny inputs otherwise.
        scale = MP_INPUT_SIZE / max(crop.shape[:2])
        resized = cv2.resize(
            crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return False
        rh, rw = resized.shape[:2]
        lm = np.array([[p.x * rw, p.y * rh, p.z * rw] for p in res.face_landmarks[0]], np.float32)
        lm[:, :2] = lm[:, :2] / scale + np.array([x1, y1], np.float32)
        lm[:, 2] /= scale
        det.landmarks = lm
        det.kps5 = kps5_from_mediapipe(lm)
        det.source = "mediapipe"
        if res.facial_transformation_matrixes:
            det.yaw, det.pitch = pose_from_matrix(np.asarray(res.facial_transformation_matrixes[0]))
        else:
            det.yaw, det.pitch = pose_from_kps5(det.kps5)
        return True
