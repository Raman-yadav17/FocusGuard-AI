"""
FocusGuard AI - web frontend
Run:  streamlit run app.py
"""
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import date, datetime

import pandas as pd
import streamlit as st

import config as cfg

FOCUS_STATES = ["STUDYING", "LOOKING AT SCREEN"]
DISTRACT_STATES = ["USING MOBILE", "SLEEPING", "NOT PRESENT"]
STATE_ORDER = ["STUDYING", "LOOKING AT SCREEN", "USING MOBILE",
               "SLEEPING", "NOT PRESENT", "ON BREAK", "FACE NOT VISIBLE"]
STATE_LABELS = {
    "USING MOBILE": "📱 Using mobile",
    "SLEEPING": "😴 Sleeping",
    "NOT PRESENT": "👤 Not present",
}

st.set_page_config(page_title="FocusGuard AI", page_icon="🎯", layout="wide")


# ------------------------------------------------------------ helpers ------
def fmt_dur(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def monitor_running():
    p = st.session_state.get("proc")
    return p is not None and p.poll() is None


def find_monitor_script():
    """Find the monitor file automatically: the .py that uses config + STOP_FLAG."""
    skip = {"app.py", "config.py", "player.py"}
    for name in sorted(os.listdir(cfg.BASE_DIR)):
        if not name.lower().endswith(".py") or name.lower() in skip:
            continue
        try:
            with open(os.path.join(cfg.BASE_DIR, name), encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue
        if "load_settings" in text and "STOP_FLAG" in text:
            return os.path.join(cfg.BASE_DIR, name)
    return None


def start_monitor():
    st.session_state.start_error = None
    script = find_monitor_script()
    if script is None:
        st.session_state.start_error = (
            "Could not find the monitor script. Make sure the updated "
            "main_v5.py (with 'from config import load_settings, STOP_FLAG') "
            "is in the project folder.")
        return
    if os.path.exists(cfg.STOP_FLAG):
        os.remove(cfg.STOP_FLAG)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    out_path = os.path.join(cfg.LOG_DIR, "monitor_output.txt")
    out = open(out_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env["FOCUSGUARD_HEADLESS"] = "1"        # no separate camera window
    proc = subprocess.Popen([sys.executable, script], cwd=cfg.BASE_DIR,
                            stdout=out, stderr=subprocess.STDOUT, env=env)
    # wait until the monitor's video server is up (models take a few seconds)
    with st.spinner("Starting monitor (loading AI models)..."):
        for _ in range(180):                    # up to ~90 seconds
            if proc.poll() is not None:
                break
            try:
                socket.create_connection(("127.0.0.1", cfg.STREAM_PORT),
                                         0.3).close()
                break
            except OSError:
                time.sleep(0.5)
    if proc.poll() is not None:
        out.flush()
        with open(out_path, encoding="utf-8", errors="ignore") as f:
            tail = "".join(f.readlines()[-15:])
        st.session_state.start_error = (
            f"{os.path.basename(script)} stopped right after starting:\n\n{tail}")
    st.session_state.stream_token = int(time.time())
    st.session_state.proc = proc


def stop_monitor():
    p = st.session_state.get("proc")
    if p is None:
        return
    open(cfg.STOP_FLAG, "w").close()       # monitor sees this and exits cleanly
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.terminate()
    if os.path.exists(cfg.STOP_FLAG):
        os.remove(cfg.STOP_FLAG)


def load_log():
    if not os.path.exists(cfg.LOG_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(cfg.LOG_FILE, parse_dates=["start", "end"])
    except Exception:
        return pd.DataFrame()
    return df


def sound_files():
    os.makedirs(cfg.ALERT_DIR, exist_ok=True)
    return sorted(f for f in os.listdir(cfg.ALERT_DIR)
                  if f.lower().endswith((".mp3", ".wav", ".ogg")))


# ------------------------------------------------------------ sidebar ------
st.sidebar.title("🎯 FocusGuard AI")
st.sidebar.caption("AI-Based Student Focus Monitoring System")

c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start", use_container_width=True, disabled=monitor_running()):
    start_monitor()
if c2.button("⏹ Stop", use_container_width=True, disabled=not monitor_running()):
    stop_monitor()

if monitor_running():
    st.sidebar.success("Monitoring is RUNNING\n\nA camera window is open. "
                       "Press Stop here (or 'q' in that window) to finish.")
else:
    st.sidebar.info("Monitoring is stopped")

if st.session_state.get("start_error"):
    st.sidebar.error("Could not start monitoring")
    st.sidebar.code(st.session_state.start_error)

st.sidebar.divider()
st.sidebar.caption("Danger zone")
confirm = st.sidebar.checkbox("I want to delete the activity log")
if st.sidebar.button("🗑 Clear activity log", disabled=not confirm):
    if os.path.exists(cfg.LOG_FILE):
        os.remove(cfg.LOG_FILE)
    st.sidebar.success("Log cleared")

tab_live, tab_dash, tab_settings, tab_calib, tab_shots, tab_about = st.tabs(
    ["📹 Live", "📊 Dashboard", "⚙️ Settings", "🎯 Calibrate",
     "📸 Screenshots", "ℹ️ About"])


# --------------------------------------------------------------- live ------
def render_live():
    if not monitor_running():
        st.info("Monitoring is stopped. Press ▶ Start in the sidebar to see "
                "the live camera view here.")
        return
    if st.checkbox("🔒 Privacy mode (hide video; detection keeps running)",
                   key="privacy"):
        st.info("Video hidden. Detection, alerts and logging are still active.")
        return
    token = st.session_state.get("stream_token", 0)
    st.markdown(
        f'<img src="http://127.0.0.1:{cfg.STREAM_PORT}/video?t={token}" '
        f'style="width:100%;max-width:960px;border-radius:10px;" />',
        unsafe_allow_html=True)
    st.caption("Live camera view. The FPS value at the bottom of the video "
               "shows how fast detection is running.")


with tab_live:
    render_live()


# ---------------------------------------------------------- dashboard ------
def compute_summary(df):
    totals = df.groupby("state")["duration_sec"].sum()
    get = lambda st_: float(totals.get(st_, 0))
    focus = sum(get(x) for x in FOCUS_STATES)
    distracted = sum(get(x) for x in DISTRACT_STATES)
    score = focus / (focus + distracted) * 100 if (focus + distracted) else 0
    return totals, get, focus, distracted, score


def build_report_html(scope, totals, focus, distracted, score):
    rows = "".join(
        f"<tr><td>{name}</td><td>{fmt_dur(v)}</td></tr>"
        for name, v in totals.sort_values(ascending=False).items())
    return (
        "<html><head><meta charset='utf-8'><title>FocusGuard AI Report</title>"
        "</head><body style='font-family:Arial;max-width:700px;margin:30px auto;'>"
        "<h1>FocusGuard AI - Focus Report</h1>"
        f"<p><b>Period:</b> {scope}<br>"
        f"<b>Generated:</b> {datetime.now():%Y-%m-%d %H:%M}</p>"
        f"<h2>Focus score: {score:.0f}%</h2>"
        f"<p>Focused time: {fmt_dur(focus)}<br>"
        f"Distracted time: {fmt_dur(distracted)}</p>"
        "<table border='1' cellpadding='8' "
        "style='border-collapse:collapse;width:100%'>"
        "<tr><th align='left'>Activity</th><th align='left'>Time</th></tr>"
        f"{rows}</table>"
        "<p style='color:#777;font-size:12px'>Tip: use your browser's "
        "Print and choose Save as PDF.</p></body></html>")


def render_dashboard():
    full = load_log()
    if full.empty:
        st.info("No activity yet. Press ▶ Start in the sidebar, study (or "
                "get distracted 😄), then stop. Data will appear here.")
        return

    full["date"] = full["start"].dt.date.astype(str)
    dates = sorted(full["date"].unique(), reverse=True)
    choice = st.selectbox("Show data for", ["All time"] + dates,
                          key="dash_choice")
    df = full if choice == "All time" else full[full["date"] == choice]

    totals, get, focus, distracted, score = compute_summary(df)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🎯 Focus score", f"{score:.0f}%")
    m2.metric("📖 Focused time", fmt_dur(focus))
    m3.metric("📱 On mobile", fmt_dur(get("USING MOBILE")))
    m4.metric("😴 Sleeping", fmt_dur(get("SLEEPING")))
    m5.metric("👤 Away", fmt_dur(get("NOT PRESENT")))
    m6.metric("🍅 Breaks taken", str(int((df["state"] == "ON BREAK").sum())))

    # daily goal (always about today)
    goal_min = int(cfg.load_settings().get("daily_goal_minutes", 120))
    today_df = full[full["date"] == str(date.today())]
    today_focus = today_df[today_df["state"].isin(FOCUS_STATES)][
        "duration_sec"].sum()
    pct = min(today_focus / (goal_min * 60), 1.0)
    st.progress(pct, text=f"Today's goal: {fmt_dur(today_focus)} focused of "
                          f"{goal_min} min ({pct * 100:.0f}%)")
    if pct >= 1:
        st.success("🎉 Daily goal reached!")

    left, right = st.columns(2)
    with left:
        st.subheader("Time per activity (minutes)")
        chart = (totals.reindex([x for x in STATE_ORDER if x in totals.index])
                 / 60).round(1)
        st.bar_chart(chart)
    with right:
        st.subheader("Activity by hour (minutes)")
        by_hour = df.copy()
        by_hour["hour"] = by_hour["start"].dt.strftime("%H:00")
        pivot = (by_hour.groupby(["hour", "state"])["duration_sec"].sum()
                 .unstack(fill_value=0) / 60).round(1)
        st.bar_chart(pivot)
        focus_cols = [c for c in FOCUS_STATES if c in pivot.columns]
        if focus_cols and pivot[focus_cols].sum(axis=1).max() > 0:
            best = pivot[focus_cols].sum(axis=1).idxmax()
            st.caption(f"⭐ Most focused hour: {best}")

    # focus score trend by day
    daily = full.groupby(["date", "state"])["duration_sec"].sum().unstack(
        fill_value=0)
    f_cols = [c for c in FOCUS_STATES if c in daily.columns]
    d_cols = [c for c in DISTRACT_STATES if c in daily.columns]
    f_day = daily[f_cols].sum(axis=1) if f_cols else daily.iloc[:, 0] * 0
    d_day = daily[d_cols].sum(axis=1) if d_cols else daily.iloc[:, 0] * 0
    total_day = (f_day + d_day).replace(0, float("nan"))
    trend = (f_day / total_day * 100).round(0).dropna()
    if len(trend) >= 2:
        st.subheader("Focus score by day (%)")
        st.line_chart(trend)

    st.subheader("Recent events")
    recent = df.sort_values("start", ascending=False).head(20)[
        ["start", "end", "state", "duration_sec"]]
    st.dataframe(recent, use_container_width=True, hide_index=True)


def render_export():
    full = load_log()
    if full.empty:
        return
    full["date"] = full["start"].dt.date.astype(str)
    choice = st.session_state.get("dash_choice", "All time")
    df = full if choice == "All time" else full[full["date"] == choice]
    if df.empty:
        df, choice = full, "All time"
    totals, _, focus, distracted, score = compute_summary(df)
    st.divider()
    st.subheader("📥 Export")
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download activity log (CSV)",
                       full.drop(columns=["date"]).to_csv(index=False).encode("utf-8"),
                       "focusguard_log.csv", "text/csv")
    c2.download_button("⬇ Download report (HTML)",
                       build_report_html(choice, totals, focus, distracted,
                                         score).encode("utf-8"),
                       "focusguard_report.html", "text/html")


# live refresh every 5 seconds (needs a recent Streamlit; falls back to manual)
if hasattr(st, "fragment"):
    render_dashboard = st.fragment(run_every=5)(render_dashboard)

with tab_dash:
    render_dashboard()
    render_export()


# ----------------------------------------------------------- settings ------
with tab_settings:
    s = cfg.load_settings()
    sounds = sound_files()

    st.subheader("🔊 Alert sounds & messages")
    up = st.file_uploader("Add your own sounds (mp3 / wav / ogg)",
                          type=["mp3", "wav", "ogg"],
                          accept_multiple_files=True)
    if up and st.button("Save uploaded sounds"):
        for f in up:
            with open(os.path.join(cfg.ALERT_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.rerun()

    if not sounds:
        st.warning("No sound files found. Upload one above or put files in "
                   "the alerts folder.")
    else:
        mime = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg"}
        for state in cfg.STATES:
            st.markdown(f"**{STATE_LABELS[state]}**")
            a, b, c = st.columns([2, 1, 3])
            cur = s["sounds"][state]
            idx = sounds.index(cur) if cur in sounds else 0
            s["sounds"][state] = a.selectbox(
                "Sound", sounds, index=idx, key=f"snd_{state}")
            s["loop"][state] = b.checkbox(
                "Loop until fixed", value=s["loop"][state], key=f"loop_{state}")
            s["messages"][state] = c.text_input(
                "Popup message", value=s["messages"][state], key=f"msg_{state}")
            path = os.path.join(cfg.ALERT_DIR, s["sounds"][state])
            ext = os.path.splitext(path)[1].lower()
            with open(path, "rb") as fh:
                st.audio(fh.read(), format=mime.get(ext, "audio/mpeg"))

    st.divider()
    st.subheader("🎚 Detection sensitivity")
    t = s["thresholds"]
    x1, x2 = st.columns(2)
    t["phone_hold"] = x1.slider("Seconds of phone use before alert", 1, 10,
                                int(t["phone_hold"]))
    t["sleep_hold"] = x1.slider("Seconds of closed eyes before alert", 1, 15,
                                int(t["sleep_hold"]))
    t["absent_hold"] = x1.slider("Seconds away before alert", 3, 60,
                                 int(t["absent_hold"]))
    t["ear_threshold"] = x2.slider("Eye-closed threshold (EAR)", 0.05, 0.40,
                                   float(t["ear_threshold"]), 0.01,
                                   help="Higher = eyes count as closed more easily")
    t["look_down_ratio"] = x2.slider("Head-down threshold (studying)", 0.30,
                                     1.00, float(t["look_down_ratio"]), 0.01,
                                     help="Use the HeadRatio shown in the camera window to calibrate")

    st.divider()
    s["enable_popup"] = st.checkbox("Show popup notifications",
                                    value=s["enable_popup"])
    s["enable_screenshots"] = st.checkbox("Save screenshot on alert",
                                          value=s["enable_screenshots"])

    st.subheader("🍅 Pomodoro timer")
    pm = s["pomodoro"]
    pm["enabled"] = st.checkbox("Enable Pomodoro (focus / break cycles)",
                                value=bool(pm["enabled"]))
    p1, p2 = st.columns(2)
    pm["focus_minutes"] = p1.slider("Focus minutes", 1, 90,
                                    int(pm["focus_minutes"]))
    pm["break_minutes"] = p2.slider("Break minutes", 1, 30,
                                    int(pm["break_minutes"]))
    st.caption("During a break, distraction alerts are paused and the time is "
               "logged as ON BREAK. Tip: set 1 and 1 to test it quickly.")

    s["daily_goal_minutes"] = st.slider(
        "Daily study goal (minutes of focused time)", 15, 480,
        int(s.get("daily_goal_minutes", 120)), 15)

    if st.button("💾 Save settings", type="primary"):
        cfg.save_settings(s)
        st.success("Saved! Stop and start monitoring again to apply changes.")


# ---------------------------------------------------------- calibrate ------
def read_metrics():
    try:
        with open(cfg.METRICS_FILE, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    return d if time.time() - d.get("ts", 0) < 3 else None


def collect(prompt, seconds, key):
    """Show a countdown, then record `key` from the monitor for `seconds`."""
    ph = st.empty()
    for i in (3, 2, 1):
        ph.warning(f"Get ready...  {prompt}  (starting in {i})")
        time.sleep(1)
    ph.info(f"▶ {prompt}")
    vals = []
    bar = st.progress(0.0)
    t0 = time.time()
    while time.time() - t0 < seconds:
        d = read_metrics()
        if d and d.get(key) is not None:
            vals.append(d[key])
        bar.progress(min((time.time() - t0) / seconds, 1.0))
        time.sleep(0.1)
    bar.empty()
    ph.empty()
    return vals


with tab_calib:
    st.subheader("🎯 Auto-calibration")
    st.write(
        "Detection depends on your camera angle, lighting and face. This "
        "wizard measures **your** values in four short steps (about 30 "
        "seconds) and sets the thresholds automatically.")
    if not monitor_running():
        st.warning("Press ▶ Start in the sidebar first, then come back here.")
    else:
        if st.button("Start calibration", type="primary"):
            steps = [
                ("screen", "Look straight at the screen", 5, "head_ratio"),
                ("book", "Look down at your book / notes", 5, "head_ratio"),
                ("open", "Look at the screen with eyes OPEN", 4, "ear"),
                ("closed", "Keep your eyes CLOSED", 4, "ear"),
            ]
            res, ok = {}, True
            for name, prompt, secs, key in steps:
                vals = collect(prompt, secs, key)
                if len(vals) < 5:
                    st.error("Could not read enough face data. Make sure your "
                             "face is visible to the camera and try again.")
                    ok = False
                    break
                res[name] = statistics.median(vals)

            if ok:
                problems = []
                if res["book"] - res["screen"] < 0.03:
                    problems.append(
                        "Looking down and looking at the screen gave almost "
                        "the same head ratio. Tilt your head down more, or "
                        "move the camera so it sees your face from the front.")
                if res["open"] - res["closed"] < 0.05:
                    problems.append(
                        "Open and closed eyes gave almost the same value. "
                        "Improve the lighting on your face and retry.")
                if problems:
                    for p in problems:
                        st.error(p)
                else:
                    s = cfg.load_settings()
                    look = (res["screen"] + res["book"]) / 2
                    ear_t = (res["open"] + res["closed"]) / 2
                    s["thresholds"]["look_down_ratio"] = round(
                        min(max(look, 0.30), 1.00), 2)
                    s["thresholds"]["ear_threshold"] = round(
                        min(max(ear_t, 0.05), 0.40), 2)
                    cfg.save_settings(s)
                    st.success("Calibration saved and applied to the running "
                               "monitor (it reloads within 2 seconds).")
                st.table(pd.DataFrame({
                    "Measured": {
                        "Head ratio - screen": round(res["screen"], 3),
                        "Head ratio - book": round(res["book"], 3),
                        "EAR - eyes open": round(res["open"], 3),
                        "EAR - eyes closed": round(res["closed"], 3),
                    }}))


# -------------------------------------------------------- screenshots ------
with tab_shots:
    st.subheader("Alert snapshots")
    os.makedirs(cfg.SHOT_DIR, exist_ok=True)
    files = [os.path.join(cfg.SHOT_DIR, f) for f in os.listdir(cfg.SHOT_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort(key=os.path.getmtime, reverse=True)
    if not files:
        st.info("No screenshots yet. They are saved when an alert triggers.")
    else:
        cols = st.columns(3)
        for i, f in enumerate(files[:12]):
            cols[i % 3].image(f, caption=os.path.basename(f))


# -------------------------------------------------------------- about ------
with tab_about:
    st.subheader("AI-Based Student Focus Monitoring System")
    st.write(
        "FocusGuard AI watches the student through a webcam and detects "
        "whether they are studying, looking at the screen, using a mobile "
        "phone, sleeping, or away from the desk. When a distraction is "
        "detected it plays a customizable sound, shows a popup message, "
        "saves a screenshot, and logs everything for the dashboard.")
    st.markdown("""
**Technologies:** YOLO (phone and person detection), MediaPipe Face
Landmarker (eye closure and head pose), OpenCV, Streamlit, Pygame.

| Activity | How it is detected |
|---|---|
| Using mobile | YOLO detects a cell phone |
| Sleeping | Eyes stay closed (Eye Aspect Ratio) |
| Studying | Head tilted down, eyes open |
| Looking at screen | Face visible, head forward |
| Not present | No person or face in the frame |
""")