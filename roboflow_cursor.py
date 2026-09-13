"""Roboflow hummingbird model on recordings or live feed.

INFER_BACKEND:
  "local" — download/cache trained weights, run on this machine
  "cloud" — hosted API (uploads each frame)

SOURCE_MODE:
  "recordings" — local clips (q = next video, ESC = quit)
  "live"       — RTSP camera (q or ESC = quit)
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

# Quiet optional-model warnings from the inference package.
for _flag in (
    "QWEN_2_5_ENABLED",
    "QWEN_3_ENABLED",
    "CORE_MODEL_SAM_ENABLED",
    "CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_SAM3_ENABLED",
    "CORE_MODEL_GAZE_ENABLED",
):
    os.environ.setdefault(_flag, "False")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|fflags;+discardcorrupt+genpts"
    "|flags;low_delay"
    "|max_delay;500000"
)
# Cut FFmpeg/libav spam (h264 "error while decoding MB …") in the terminal.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")  # AV_LOG_FATAL

import cv2

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv(ROOT / ".env")

# --- settings ---
INFER_BACKEND = "cloud"  # "local" | "cloud"
SOURCE_MODE = "live"  # "recordings" | "live"
MODEL_ID = "braden-gr0sv/hummingbirds-p7kwn-1-yolo11n-t3"
# MODEL_ID = "braden-gr0sv/hummingbirds-p7kwn/1"
CONFIDENCE = 0.2
PLAYBACK_SPEED = 2.0  # recordings only
RECORD_ON_DETECT = False   # live only: save a clip around each visit
POST_DETECT_HOLD_S = 3.0  # keep recording this long after last detection; resets if seen again
DETECT_CLIP_DIR = ROOT / "dataset" / "detections"
DETECT_CLIP_FPS = 15.0  # RTSP often reports bad FPS; fixed rate for writers
RTSP_URL = "rtsp://192.168.0.242:8554/front-door-cam"
RECORDINGS = sorted(
    p for p in DETECT_CLIP_DIR.iterdir()
    if p.suffix.lower() in {".mp4", ".mov"}
    # if p.prefix.lower() in {'detect_'}
)
WINDOW = "roboflow test"
# ----------------

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY missing — set it in .env")


def _filter_preds(raw) -> list[dict]:
    """Normalize Roboflow / SDK prediction objects or dicts → draw-friendly dicts."""
    preds = []
    for p in raw:
        if isinstance(p, dict):
            conf = float(p.get("confidence", 0))
            if conf < CONFIDENCE:
                continue
            preds.append(
                {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "width": float(p["width"]),
                    "height": float(p["height"]),
                    "confidence": conf,
                    "class": str(p.get("class") or p.get("class_name") or "?"),
                }
            )
            continue
        conf = float(p.confidence)
        if conf < CONFIDENCE:
            continue
        preds.append(
            {
                "x": float(p.x),
                "y": float(p.y),
                "width": float(p.width),
                "height": float(p.height),
                "confidence": conf,
                "class": str(getattr(p, "class_name", None) or getattr(p, "class", "?")),
            }
        )
    return preds


def make_cloud_detect():
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    def detect(frame):
        result = client.infer(frame, model_id=MODEL_ID)
        return _filter_preds(result.get("predictions", []))

    return detect


def make_local_detect():
    from inference import get_model

    model = get_model(model_id=MODEL_ID, api_key=api_key)

    def detect(frame):
        result = model.infer(frame, confidence=CONFIDENCE)[0]
        return _filter_preds(result.predictions)

    return detect


def build_detect_fn():
    if INFER_BACKEND == "cloud":
        return make_cloud_detect()
    if INFER_BACKEND == "local":
        return make_local_detect()
    raise SystemExit(f'INFER_BACKEND must be "local" or "cloud", got {INFER_BACKEND!r}')


class AsyncDetector:
    """Run detect() off the UI thread; keep last preds for drawing."""

    def __init__(self, detect_fn):
        self._detect = detect_fn
        self._lock = threading.Lock()
        self._busy = False
        self._preds: list[dict] = []

    def maybe_submit(self, frame, label: str, frame_i: int) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(
            target=self._run,
            args=(frame.copy(), label, frame_i),
            daemon=True,
        ).start()

    def _run(self, frame, label: str, frame_i: int) -> None:
        try:
            preds = self._detect(frame)
            if preds:
                print(
                    f"  [{label}] frame {frame_i}:",
                    [(p["class"], round(p["confidence"], 2)) for p in preds],
                )
            with self._lock:
                self._preds = preds
        except Exception as exc:
            print(f"infer error: {exc}")
        finally:
            with self._lock:
                self._busy = False

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._preds)


def draw_preds(frame, preds: list[dict], title: str, *, recording: bool = False):
    out = frame.copy()
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if recording:
        cv2.circle(out, (out.shape[1] - 24, 24), 10, (0, 0, 255), -1)
        cv2.putText(out, "REC", (out.shape[1] - 78, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for p in preds:
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        tag = f"{p['class']} {p['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    return out


class DetectClipRecorder:
    """Start on detection; keep going POST_DETECT_HOLD_S after it vanishes; reset hold if seen again."""

    def __init__(self, out_dir: Path, fps: float, hold_s: float):
        self.out_dir = out_dir
        self.fps = fps
        self.hold_s = hold_s
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._stop_deadline: float | None = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    def update(self, frame, seen: bool) -> None:
        now = time.monotonic()
        if seen:
            self._stop_deadline = None
            if self._writer is None:
                self._start(frame)
        elif self._writer is not None:
            if self._stop_deadline is None:
                self._stop_deadline = now + self.hold_s
            elif now >= self._stop_deadline:
                self.stop()
                return

        if self._writer is not None:
            self._writer.write(frame)

    def _start(self, frame) -> None:
        h, w = frame.shape[:2]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._path = self.out_dir / f"detect_{stamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, self.fps, (w, h))
        if not self._writer.isOpened():
            print(f"failed to open writer: {self._path}")
            self._writer = None
            self._path = None
            return
        print(f"REC start → {self._path.name}")

    def stop(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        print(f"REC stop  → {self._path.name if self._path else '?'}")
        self._path = None
        self._stop_deadline = None


def open_rtsp(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"Could not open RTSP stream: {url}")
    # Small buffer: stay near-live, but >1 so the H.264 decoder still has refs.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


def run_live(label: str, cap: cv2.VideoCapture, detect_fn) -> None:
    """Real-time RTSP loop — no frame-skip scheduling (that stalls live streams)."""
    detector = AsyncDetector(detect_fn)
    recorder = (
        DetectClipRecorder(DETECT_CLIP_DIR, DETECT_CLIP_FPS, POST_DETECT_HOLD_S)
        if RECORD_ON_DETECT
        else None
    )
    frame_i = 0
    fail_streak = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fail_streak += 1
                if fail_streak >= 30:
                    print("RTSP read failing — reconnecting…")
                    cap.release()
                    time.sleep(1.0)
                    cap.open(RTSP_URL, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    fail_streak = 0
                continue
            fail_streak = 0

            detector.maybe_submit(frame, label, frame_i)
            preds = detector.snapshot()
            if recorder is not None:
                recorder.update(frame, seen=bool(preds))

            title = label
            if recorder is not None and recorder.active:
                title = f"{label}  REC"
            cv2.imshow(
                WINDOW,
                draw_preds(frame, preds, title, recording=bool(recorder and recorder.active)),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            frame_i += 1
    finally:
        if recorder is not None:
            recorder.stop()


def run_recordings(
    label: str,
    cap: cv2.VideoCapture,
    detect_fn,
    *,
    fps: float,
    playback_speed: float,
) -> bool:
    """Play file at playback_speed (wall-clock); skip frames if behind. True = ESC."""
    detector = AsyncDetector(detect_fn)
    frame_i = 0
    period = 1.0 / max(fps * playback_speed, 1e-3)
    next_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detector.maybe_submit(frame, label, frame_i)
        cv2.imshow(WINDOW, draw_preds(frame, detector.snapshot(), label))

        now = time.perf_counter()
        while now > next_t + period:
            if not cap.grab():
                return False
            frame_i += 1
            next_t += period
            now = time.perf_counter()

        wait_ms = max(1, int((next_t - now) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        next_t += period
        if key == 27:
            return True
        if key == ord("q"):
            break
        frame_i += 1

    return False


def main() -> None:
    if SOURCE_MODE not in ("recordings", "live"):
        raise SystemExit(f'SOURCE_MODE must be "recordings" or "live", got {SOURCE_MODE!r}')

    detect_fn = build_detect_fn()
    print(f"Backend={INFER_BACKEND} | mode={SOURCE_MODE} | model={MODEL_ID}")

    if SOURCE_MODE == "live":
        extras = f" | record_on_detect={RECORD_ON_DETECT} hold={POST_DETECT_HOLD_S:g}s"
        print(f"Live: {RTSP_URL} | q/ESC=quit{extras}")
        if RECORD_ON_DETECT:
            print(f"Clips → {DETECT_CLIP_DIR}/")
        cap = open_rtsp(RTSP_URL)
        try:
            run_live("live", cap, detect_fn)
        finally:
            cap.release()
    else:
        print(f"Recordings @ {PLAYBACK_SPEED:g}x | q=next video, ESC=quit")
        for path in RECORDINGS:
            if not path.exists():
                print(f"skip missing: {path.name}")
                continue
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print(f"could not open: {path.name}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps < 1:
                fps = 30.0
            print(f"\n=== {path.name} ({fps:.1f} fps @ {PLAYBACK_SPEED:g}x) ===")
            try:
                stop = run_recordings(
                    f"{path.name}  {PLAYBACK_SPEED:g}x",
                    cap,
                    detect_fn,
                    fps=fps,
                    playback_speed=PLAYBACK_SPEED,
                )
            finally:
                cap.release()
            if stop:
                break

    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    main()
