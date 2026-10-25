"""Shared paths and settings for the monitor (main_v5.py) and the web app (app.py)."""
import copy
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERT_DIR = os.path.join(BASE_DIR, "alerts")
SHOT_DIR = os.path.join(BASE_DIR, "screenshots")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "activity_log.csv")
CONFIG_FILE = os.path.join(BASE_DIR, "settings.json")
STOP_FLAG = os.path.join(BASE_DIR, "stop.flag")
LIVE_FRAME = os.path.join(BASE_DIR, "live_frame.jpg")      # latest camera frame
LIVE_TMP = os.path.join(BASE_DIR, "live_frame_tmp.jpg")
METRICS_FILE = os.path.join(BASE_DIR, "live_metrics.json")  # live EAR / head ratio
STREAM_PORT = 8765                                          # live video stream

# States that can trigger an alert
STATES = ["USING MOBILE", "SLEEPING", "NOT PRESENT"]

DEFAULT_SETTINGS = {
    "sounds": {
        "USING MOBILE": "alert.mp3",
        "SLEEPING": "alert.mp3",
        "NOT PRESENT": "universfield-notification-beep-229154.mp3",
    },
    "loop": {"USING MOBILE": True, "SLEEPING": True, "NOT PRESENT": False},
    "messages": {
        "USING MOBILE": "Put the phone away and focus!",
        "SLEEPING": "Wake up! Time to study.",
        "NOT PRESENT": "You left your desk. Come back to study.",
    },
    "daily_goal_minutes": 120,
    "pomodoro": {"enabled": False, "focus_minutes": 25, "break_minutes": 5},
    "enable_popup": True,
    "enable_screenshots": True,
    "thresholds": {
        "phone_hold": 2,         # sec phone must be seen before alert
        "sleep_hold": 4,         # sec eyes closed before alert
        "absent_hold": 8,        # sec absent before alert
        "ear_threshold": 0.21,   # below = eyes closed
        "look_down_ratio": 0.62, # above = head tilted down
    },
}


def load_settings():
    """Defaults, overridden by whatever is saved in settings.json."""
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key, value in saved.items():
                if isinstance(value, dict) and isinstance(settings.get(key), dict):
                    settings[key].update(value)
                else:
                    settings[key] = value
        except Exception as e:
            print("Could not read settings.json, using defaults:", e)
    return settings


def save_settings(settings):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)