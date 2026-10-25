import csv
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from config import (load_settings, STOP_FLAG, STREAM_PORT,
                    CONFIG_FILE, METRICS_FILE)

# The web app sets this so the monitor does not open its own window
HEADLESS = os.environ.get("FOCUSGUARD_HEADLESS") == "1"

# ============================ PATHS ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo11n.pt")
FACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/"
                  "face_landmarker.task")
PLAYER_SCRIPT = os.path.join(BASE_DIR, "player.py")
SHOT_DIR = os.path.join(BASE_DIR, "screenshots")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "activity_log.csv")
os.makedirs(SHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

ALERT_MUSIC = os.path.join(BASE_DIR, "alerts", "alert.mp3")
ALERT_BEEP = os.path.join(BASE_DIR, "alerts",
                          "universfield-notification-beep-229154.mp3")

# ============ SETTINGS (edit in the web app, or in settings.json) ============
SETTINGS = load_settings()
ACTIONS = {
    state: {
        "sound": os.path.join(BASE_DIR, "alerts", SETTINGS["sounds"][state]),
        "loop": SETTINGS["loop"][state],
        "message": SETTINGS["messages"][state],
    }
    for state in SETTINGS["sounds"]
}
ENABLE_POPUP = SETTINGS["enable_popup"]
ENABLE_SCREENSHOTS = SETTINGS["enable_screenshots"]
NOTIFY_COOLDOWN = 20     # sec between popups/screenshots for the same state

T = SETTINGS["thresholds"]
PHONE_ID, PERSON_ID = 67, 0
PHONE_CONF = 0.5
PHONE_HOLD = T["phone_hold"]
SLEEP_HOLD = T["sleep_hold"]
ABSENT_HOLD = T["absent_hold"]
STUDY_HOLD = 1          # sec head must stay down to count as studying
ALERT_CLEAR = 3         # sec of "all good" before looping alert stops

EAR_THRESHOLD = T["ear_threshold"]
EAR_DOWN_FACTOR = 0.80  # stricter when head is down (eyelids look narrow)
LOOK_DOWN_RATIO = T["look_down_ratio"]
SHOW_DEBUG = True

# ---- speed settings (lower YOLO_IMGSZ / raise YOLO_EVERY = faster) ----
YOLO_IMGSZ = 416        # YOLO input size: 320 (fast) ... 640 (accurate)
YOLO_EVERY = 2          # run YOLO on every Nth frame
CAMERA_W, CAMERA_H = 640, 480
JPEG_QUALITY = 75

# ---- Pomodoro (focus / break cycles) ----
_PM = SETTINGS["pomodoro"]
POMO_ON = _PM["enabled"]
POMO_FOCUS_S = int(_PM["focus_minutes"]) * 60
POMO_BREAK_S = int(_PM["break_minutes"]) * 60

# remove a stale stop request from a previous run
if os.path.exists(STOP_FLAG):
    os.remove(STOP_FLAG)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE, FOREHEAD, CHIN = 1, 10, 152


# ============================ HELPERS ============================
def ensure_face_model():
    if os.path.exists(FACE_MODEL_PATH):
        return
    print("Downloading face_landmarker.task (one time)...")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    except Exception as e:
        print("Download failed:", e)
        print("Download manually from:\n", FACE_MODEL_URL)
        print("and save it as:", FACE_MODEL_PATH)
        sys.exit()


def eye_aspect_ratio(lm, idx, w, h):
    p = [np.array([lm[i].x * w, lm[i].y * h]) for i in idx]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    hz = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * hz + 1e-6)


def head_down_ratio(lm):
    top, nose, chin = lm[FOREHEAD].y, lm[NOSE].y, lm[CHIN].y
    return (nose - top) / (chin - top + 1e-6)


class Hold:
    """True only after a condition stays true for `hold` seconds.
    Short flickers (<= gap seconds) do not reset the timer."""

    def __init__(self, hold, gap=0.7):
        self.hold, self.gap = hold, gap
        self.since, self.last = None, 0.0

    def update(self, cond, now):
        if cond:
            self.last = now
            if self.since is None:
                self.since = now
        elif self.since is not None and now - self.last > self.gap:
            self.since = None
        return self.since is not None and now - self.since >= self.hold


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ============================ SOUND ============================
alert_process = None
alert_is_loop = False


def alert_running():
    return alert_process is not None and alert_process.poll() is None


def stop_alert():
    global alert_process
    if alert_running():
        alert_process.terminate()
    alert_process = None


def start_alert(sound, loop):
    global alert_process, alert_is_loop
    stop_alert()
    if not os.path.exists(sound):
        print("Sound file missing:", sound)
        return
    cmd = [sys.executable, PLAYER_SCRIPT, sound]
    if loop:
        cmd.append("--loop")
    alert_process = subprocess.Popen(cmd)
    alert_is_loop = loop


# ============================ POPUP ============================
def notify(msg):
    if not ENABLE_POPUP:
        return

    def _run():
        try:
            from plyer import notification
            notification.notify(title="AI Study Monitor", message=msg,
                                timeout=5)
        except Exception:
            print("[NOTIFY]", msg)

    threading.Thread(target=_run, daemon=True).start()


# ============================ LOGGING ============================
totals = defaultdict(float)


def log_segment(state, start, end):
    dur = end - start
    if dur < 0.5:
        return
    totals[state] += dur
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["start", "end", "state", "duration_sec"])
        w.writerow([datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
                    state, round(dur, 1)])


def print_summary():
    print("\n===== SESSION SUMMARY =====")
    if not totals:
        print("No activity recorded.")
    for state, sec in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"{state:<20} {fmt_time(sec)}")
    print(f"Log saved to: {LOG_FILE}")


# ====================== LIVE VIDEO STREAM (MJPEG) ======================
class FrameBuffer:
    """Holds the newest JPEG so the stream server can send it."""

    def __init__(self):
        self.jpeg, self.id = None, 0
        self.lock = threading.Lock()

    def set(self, jpeg_bytes):
        with self.lock:
            self.jpeg, self.id = jpeg_bytes, self.id + 1

    def get(self):
        with self.lock:
            return self.jpeg, self.id


frames = FrameBuffer()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):       # keep the terminal quiet
        pass

    def do_GET(self):
        if not self.path.startswith("/video"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        last = -1
        try:
            while True:
                jpg, fid = frames.get()
                if jpg is not None and fid != last:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpg)).encode() +
                        b"\r\n\r\n" + jpg + b"\r\n")
                    last = fid
                time.sleep(0.02)
        except OSError:                 # browser closed / tab changed
            pass


def start_stream_server():
    try:
        server = ThreadingHTTPServer(("127.0.0.1", STREAM_PORT), StreamHandler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Live stream: http://127.0.0.1:{STREAM_PORT}/video")
    except OSError as e:
        print("Could not start live stream server:", e)


# ======================== THREADED CAMERA READER ========================
class CameraStream:
    """Reads the webcam in a background thread so frames never queue up."""

    def __init__(self, index=0):
        self.cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        self.opened = self.cap is not None
        self.frame, self.id, self.failed = None, 0, False
        self.running = self.opened
        self.lock = threading.Lock()
        if self.opened:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.failed = True
                time.sleep(0.05)
                continue
            self.failed = False
            with self.lock:
                self.frame, self.id = frame, self.id + 1

    def read(self):
        with self.lock:
            return self.frame, self.id

    def release(self):
        self.running = False
        time.sleep(0.1)
        if self.cap is not None:
            self.cap.release()


# ============================ SETUP ============================
model = YOLO(MODEL_PATH)
ensure_face_model()
face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5))

camera = CameraStream(0)
if not camera.opened:
    print("Camera not found (is another program using it?)")
    sys.exit()

phone_hold = Hold(PHONE_HOLD, gap=1.0)
sleep_hold = Hold(SLEEP_HOLD, gap=0.7)
absent_hold = Hold(ABSENT_HOLD, gap=0.5)
study_hold = Hold(STUDY_HOLD, gap=1.0)

LOOP_STATES = {s for s, c in ACTIONS.items() if c["loop"]}
start_time = time.time()
last_ts = -1
last_bad = 0.0
last_metrics_write = 0.0
pomo_phase, pomo_start, pomo_sessions = "FOCUS", time.time(), 0
last_settings_check = 0.0
settings_mtime = (os.path.getmtime(CONFIG_FILE)
                  if os.path.exists(CONFIG_FILE) else 0)
last_notified = {}
current_state, state_start = None, time.time()

frame_count = 0
last_id = -1
last_boxes = []                 # detections reused between YOLO runs
phone_seen = person_seen = False
fps, t_prev = 0.0, time.time()

start_stream_server()           # web app waits for this before showing video

# ============================= LOOP =============================
try:
    while True:
        frame, fid = camera.read()
        if frame is None or fid == last_id:
            if camera.failed:
                print("Camera error")
                break
            if os.path.exists(STOP_FLAG):
                break
            time.sleep(0.003)
            continue
        last_id = fid

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        h, w = frame.shape[:2]
        frame_count += 1

        # re-read thresholds when settings.json changes (calibration / sliders)
        if now - last_settings_check > 2:
            last_settings_check = now
            try:
                mt = os.path.getmtime(CONFIG_FILE)
                if mt != settings_mtime:
                    settings_mtime = mt
                    T = load_settings()["thresholds"]
                    EAR_THRESHOLD = T["ear_threshold"]
                    LOOK_DOWN_RATIO = T["look_down_ratio"]
                    phone_hold.hold = T["phone_hold"]
                    sleep_hold.hold = T["sleep_hold"]
                    absent_hold.hold = T["absent_hold"]
                    PM = load_settings()["pomodoro"]
                    was_on = POMO_ON
                    POMO_ON = PM["enabled"]
                    POMO_FOCUS_S = int(PM["focus_minutes"]) * 60
                    POMO_BREAK_S = int(PM["break_minutes"]) * 60
                    if POMO_ON and not was_on:
                        pomo_phase, pomo_start = "FOCUS", now
                    print("Settings reloaded")
            except OSError:
                pass

        # 1) YOLO (phone + person) on every Nth frame, reuse in between
        if frame_count % YOLO_EVERY == 0:
            results = model(frame, verbose=False, imgsz=YOLO_IMGSZ,
                            classes=[PERSON_ID, PHONE_ID], conf=0.4)
            phone_seen = person_seen = False
            last_boxes = []
            for box in results[0].boxes:
                cid, conf = int(box.cls[0]), float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if cid == PHONE_ID and conf > PHONE_CONF:
                    phone_seen = True
                    last_boxes.append((x1, y1, x2, y2,
                                       f"phone {conf:.2f}", (0, 0, 255)))
                elif cid == PERSON_ID:
                    person_seen = True
                    last_boxes.append((x1, y1, x2, y2,
                                       f"person {conf:.2f}", (0, 200, 0)))

        # 2) MediaPipe: eyes + head pose (every frame, it is light)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((now - start_time) * 1000), last_ts + 1)
        last_ts = ts
        out = face_landmarker.detect_for_video(mp_image, ts)
        face_seen = len(out.face_landmarks) > 0
        ear = down = None
        eyes_closed = head_down = False
        if face_seen:
            lm = out.face_landmarks[0]
            ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) +
                   eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2
            down = head_down_ratio(lm)
            head_down = down > LOOK_DOWN_RATIO
            thr = EAR_THRESHOLD * (EAR_DOWN_FACTOR if head_down else 1.0)
            eyes_closed = ear < thr

        # 3) Decide state
        phone_ok = phone_hold.update(phone_seen, now)
        sleep_ok = sleep_hold.update(eyes_closed, now)
        absent_ok = absent_hold.update(not person_seen and not face_seen, now)
        study_ok = study_hold.update(head_down and not eyes_closed, now)

        if phone_ok:
            state, color = "USING MOBILE", (0, 0, 255)
        elif sleep_ok:
            state, color = "SLEEPING", (0, 0, 255)
        elif absent_ok:
            state, color = "NOT PRESENT", (0, 165, 255)
        elif study_ok:
            state, color = "STUDYING", (0, 200, 0)
        elif face_seen:
            state, color = "LOOKING AT SCREEN", (0, 200, 0)
        else:
            state, color = "FACE NOT VISIBLE", (200, 200, 0)

        # 3b) Pomodoro: focus / break cycle
        pomo_text = None
        if POMO_ON:
            phase_len = POMO_FOCUS_S if pomo_phase == "FOCUS" else POMO_BREAK_S
            if now - pomo_start >= phase_len:
                if pomo_phase == "FOCUS":
                    pomo_phase, pomo_sessions = "BREAK", pomo_sessions + 1
                    msg = f"Great work! Take a {POMO_BREAK_S // 60} minute break."
                else:
                    pomo_phase = "FOCUS"
                    msg = "Break over! Back to studying."
                pomo_start = now
                phase_len = POMO_FOCUS_S if pomo_phase == "FOCUS" else POMO_BREAK_S
                start_alert(ALERT_BEEP, False)      # replaces any running alert
                notify(msg)
                print("🍅", msg)
            remaining = max(phase_len - (now - pomo_start), 0)
            pomo_text = (f"{pomo_phase} {fmt_time(remaining)}  |  "
                         f"Sessions: {pomo_sessions}")
            if pomo_phase == "BREAK":               # no distraction alerts on break
                state, color = "ON BREAK", (255, 160, 0)

        # 4) Draw (directly on the frame - much faster than results.plot())
        annotated = frame
        for (x1, y1, x2, y2, label, col) in last_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(annotated, state, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3)
        if pomo_text:
            cv2.putText(annotated, pomo_text, (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255) if pomo_phase == "FOCUS" else (255, 160, 0),
                        2)
        live = dict(totals)
        live[state] = live.get(state, 0) + (now - state_start)
        y = 85
        for name in ("STUDYING", "LOOKING AT SCREEN", "USING MOBILE",
                     "SLEEPING", "NOT PRESENT"):
            cv2.putText(annotated, f"{name}: {fmt_time(live.get(name, 0))}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)
            y += 24
        if SHOW_DEBUG:
            txt = (f"EAR: {ear:.2f}  HeadRatio: {down:.2f}  FPS: {fps:.0f}"
                   if face_seen else f"No face landmarks  FPS: {fps:.0f}")
            cv2.putText(annotated, txt, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5) State changed -> log previous, trigger alert for the new one
        if state != current_state:
            if current_state is not None:
                log_segment(current_state, state_start, now)
            current_state, state_start = state, now

            cfg = ACTIONS.get(state)
            if cfg:
                start_alert(cfg["sound"], cfg["loop"])
                if now - last_notified.get(state, 0) > NOTIFY_COOLDOWN:
                    last_notified[state] = now
                    notify(cfg["message"])
                    if ENABLE_SCREENSHOTS:
                        name = (f"{state.replace(' ', '_')}_"
                                f"{datetime.now():%Y%m%d_%H%M%S}.jpg")
                        cv2.imwrite(os.path.join(SHOT_DIR, name), annotated)
                    print(f"⚠️  {state}: {cfg['message']}")

        # 6) Stop looping alert shortly after behavior is fixed
        if state in LOOP_STATES:
            last_bad = now
        elif alert_running() and alert_is_loop and now - last_bad > ALERT_CLEAR:
            stop_alert()

        # 6b) live numbers for the web app's calibration wizard
        if now - last_metrics_write > 0.3:
            last_metrics_write = now
            try:
                with open(METRICS_FILE + ".tmp", "w", encoding="utf-8") as f:
                    json.dump({
                        "ts": now, "state": state, "fps": round(fps, 1),
                        "ear": None if ear is None else round(float(ear), 4),
                        "head_ratio": None if down is None else round(float(down), 4),
                    }, f)
                os.replace(METRICS_FILE + ".tmp", METRICS_FILE)
            except OSError:
                pass

        # 7) Send frame to the web app's live video
        ok, buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            frames.set(buf.tobytes())

        if not HEADLESS:
            cv2.imshow("AI Study Monitor", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if os.path.exists(STOP_FLAG):      # Stop button in the web app
            break

finally:
    if current_state is not None:
        log_segment(current_state, state_start, time.time())
    stop_alert()
    try:
        os.remove(METRICS_FILE)
    except OSError:
        pass
    camera.release()
    cv2.destroyAllWindows()
    print_summary()