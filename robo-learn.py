import os
from pathlib import Path

import cv2
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

# load .env
env = Path(__file__).resolve().parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY missing")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key,
)
client.configure(InferenceConfiguration(api_key_transport="header"))

cap = cv2.VideoCapture("rtsp://192.168.0.242:8554/front-door-cam", cv2.CAP_FFMPEG) # LIVE
# cap = cv2.VideoCapture("./dataset/recordings/detect_098.mp4")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    results = client.run_workflow(
        workspace_name="braden-gr0sv",
        workflow_id="hummingbirds-p7kwn",
        images={"image": frame},
    )
    # print(results[0] if results else results)

    # cv2.imshow("hummingbird", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
