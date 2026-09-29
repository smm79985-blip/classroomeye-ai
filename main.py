"""
ClassroomEye — AI Engagement Detection Service
FastAPI microservice: accepts webcam frames, runs MediaPipe FaceMesh,
returns per-student engagement scores (0-100) with pose + eye data.
"""

import base64
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("classroomeye.ai")

mp_face_mesh = mp.solutions.face_mesh
_executor = ThreadPoolExecutor(max_workers=8)

app = FastAPI(title="ClassroomEye AI Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])

class FrameRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded image")
    student_id: Optional[str] = None

    @validator("frame")
    def must_be_nonempty(cls, v):
        if not v or len(v) < 100:
            raise ValueError("frame must be a non-empty base64 string")
        return v

class BatchFrameRequest(BaseModel):
    frames: list[FrameRequest] = Field(..., min_items=1, max_items=30)

class EngagementResult(BaseModel):
    student_id: Optional[str]
    engagement_score: float
    yaw: float
    pitch: float
    roll: float
    eye_openness: float
    is_present: bool
    confidence: float
    processing_ms: float
    error: Optional[str] = None

class BatchEngagementResult(BaseModel):
    results: list[EngagementResult]
    total_processing_ms: float

POSE_LANDMARK_IDS = [1, 33, 263, 61, 291, 199]
LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 159, 145, 33, 133
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 386, 374, 362, 263

MODEL_3D_POINTS = np.array([
    [0.0, 0.0, 0.0], [-30.0, -30.0, -30.0], [30.0, -30.0, -30.0],
    [-20.0, 20.0, -20.0], [20.0, 20.0, -20.0], [0.0, 55.0, -15.0],
], dtype=np.float64)

def _decode_frame(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def _preprocess(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def _estimate_head_pose(landmarks, img_w, img_h):
    image_points = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in POSE_LANDMARK_IDS], dtype=np.float64)
    focal_length = img_w
    camera_matrix = np.array([[focal_length, 0, img_w/2], [0, focal_length, img_h/2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy >= 1e-6:
        roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    else:
        roll = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw = 0.0
    confidence = float(np.mean([landmarks[i].visibility for i in POSE_LANDMARK_IDS]))
    return yaw, pitch, roll, confidence

def _eye_aspect_ratio(landmarks, img_w, img_h):
    def dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    lv = dist(landmarks[LEFT_EYE_TOP], landmarks[LEFT_EYE_BOTTOM])
    lh = dist(landmarks[LEFT_EYE_LEFT], landmarks[LEFT_EYE_RIGHT])
    rv = dist(landmarks[RIGHT_EYE_TOP], landmarks[RIGHT_EYE_BOTTOM])
    rh = dist(landmarks[RIGHT_EYE_LEFT], landmarks[RIGHT_EYE_RIGHT])
    ear = ((lv / (lh + 1e-6)) + (rv / (rh + 1e-6))) / 2.0
    return round(min(1.0, ear / 0.30), 4)

def _compute_engagement(yaw, pitch, roll, eye_openness, is_present, confidence):
    if not is_present:
        return 0.0
    def angle_score(angle, soft, hard):
        a = abs(angle)
        if a <= soft: return 1.0
        if a >= hard: return 0.0
        return 1.0 - (a - soft) / (hard - soft)
    pose_score = 0.60 * angle_score(yaw, 15, 45) + 0.30 * angle_score(pitch, 10, 35) + 0.10 * angle_score(roll, 20, 50)
    presence_score = min(1.0, confidence * 1.2)
    raw = (0.40 * pose_score + 0.30 * eye_openness + 0.30 * presence_score) * 100
    return round(min(100.0, max(0.0, raw)), 2)

def _process_single_frame(req: FrameRequest) -> EngagementResult:
    t0 = time.perf_counter()
    try:
        img_bgr = _decode_frame(req.frame)
        img_bgr = _preprocess(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(img_rgb)
        is_present = bool(results.multi_face_landmarks)
        if not is_present:
            elapsed = (time.perf_counter() - t0) * 1000
            return EngagementResult(student_id=req.student_id, engagement_score=0.0, yaw=0.0, pitch=0.0, roll=0.0, eye_openness=0.0, is_present=False, confidence=0.0, processing_ms=round(elapsed, 2))
        landmarks = results.multi_face_landmarks[0].landmark
        yaw, pitch, roll, confidence = _estimate_head_pose(landmarks, w, h)
        eye_openness = _eye_aspect_ratio(landmarks, w, h)
        score = _compute_engagement(yaw, pitch, roll, eye_openness, True, confidence)
        elapsed = (time.perf_counter() - t0) * 1000
        return EngagementResult(student_id=req.student_id, engagement_score=score, yaw=round(yaw, 2), pitch=round(pitch, 2), roll=round(roll, 2), eye_openness=eye_openness, is_present=True, confidence=round(confidence, 4), processing_ms=round(elapsed, 2))
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return EngagementResult(student_id=req.student_id, engagement_score=0.0, yaw=0.0, pitch=0.0, roll=0.0, eye_openness=0.0, is_present=False, confidence=0.0, processing_ms=round(elapsed, 2), error=str(exc))

@app.get("/health")
def health():
    return {"status": "ok", "service": "classroomeye-ai"}

@app.post("/analyze", response_model=EngagementResult)
def analyze_frame(req: FrameRequest):
    return _process_single_frame(req)

@app.post("/analyze/batch", response_model=BatchEngagementResult)
def analyze_batch(req: BatchFrameRequest):
    t0 = time.perf_counter()
    futures = [_executor.submit(_process_single_frame, f) for f in req.frames]
    results = [f.result() for f in futures]
    return BatchEngagementResult(results=results, total_processing_ms=round((time.perf_counter() - t0) * 1000, 2))
