"""Record RTSP forever into ~15-minute H.264 MP4 files. Ctrl+C to stop."""
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

RTSP = "rtsp://192.168.0.242:8554/front-door-cam"
OUT = Path(__file__).resolve().parent / "dataset" / "recordings"
SEGMENT_S = 15 * 60
MIN_KEEP_S = 30  # keep partials at least this long; delete tiny failures

OUT.mkdir(parents=True, exist_ok=True)
ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
if not ffmpeg:
    raise SystemExit("ffmpeg not found. Install with: brew install ffmpeg")

print(f"Recording {RTSP}")
print(f"Target ~{SEGMENT_S // 60}-min H.264 MP4 files -> {OUT}/")
print("Reconnects automatically if the stream drops. Ctrl+C to stop.\n")


def duration_s(path: Path) -> float:
    if not ffprobe or not path.exists():
        return 0.0
    r = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


while True:
    out_path = OUT / f"rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "warning",
        "-rtsp_transport", "tcp",
        "-rtsp_flags", "prefer_tcp",
        "-allowed_media_types", "video",
        "-fflags", "+genpts+igndts",
        "-i", RTSP,
        "-t", str(SEGMENT_S),
        "-map", "0:v:0",
        "-c:v", "copy",
        "-tag:v", "avc1",
        "-an",
        "-f", "mp4",
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
        "-y",
        str(out_path),
    ]
    print(f"Writing {out_path.name} ...")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\nStopped.")
        sys.exit(0)

    secs = duration_s(out_path)
    if secs < MIN_KEEP_S:
        out_path.unlink(missing_ok=True)
        print(f"Stream dropped too soon ({secs:.1f}s); retrying in 5s...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped.")
            sys.exit(0)
        continue

    if secs < SEGMENT_S - 5:
        print(f"Stream ended early — kept {out_path.name} ({secs / 60:.1f} min). Reconnecting...")
    else:
        print(f"Finished {out_path.name} ({secs / 60:.1f} min)")
