import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
import cv2
from ultralytics import YOLO
from pathlib import Path
import time

rtsp_url = 'rtsp://192.168.0.242:8554/front-door-cam'
model = YOLO('yolov26n.pt')
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

out_train = Path(__file__).resolve().parent / "dataset" / "images" / "train"
out_train.mkdir(parents=True, exist_ok=True)
last_save=0
interval_s = 2.0
save_images = False

def save_img(name="",interval=interval_s):
    if save_images is not True:
        return
    now = time.time()
    name = name or now
    global last_save
    print(now-last_save > interval)
    if now-last_save > interval:
        path = out_train / f"{now}.jpg"
        cv2.imwrite(str(path), frame)
        last_save = now
        print(f'saved {now}')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # results = model(frame, verbose=False)
    # if results[0].boxes is not None and len(results[0].boxes):
    #     print(results[0].boxes)  # or whatever summary you want



    results = model.track(
        frame,
        persist=True,
        verbose=False,
        conf=0.10,
        imgsz=1280,
        classes=[0,2,14],
        # tracker="bytetrack.yaml",
    )

    r = results[0]
    if r.boxes.id is not None:
        # save_img()
        print(r.boxes.id.int().cpu().tolist(), r.boxes.cls.int().cpu().tolist())
    if results[0]:
        save_img()
    # results = model.track(source=frame, verbose=False, persist=True)
    # for result in results:
    # # Use is_track attribute to check if tracking data exists
    #     if result.boxes.is_track:
    #         # Safely extract tracking IDs
    #         track_ids = result.boxes.id.int().cpu().tolist()
    #         print(f"Active tracking IDs in this frame: {track_ids}")
    #     else:
    #         print("Standard object detection results only—no tracking IDs found.")



    annotated_frame = results[0].plot()
    cv2.imshow('obj detect', annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("a"):
        save_img()
    if key == ord("q"):
        break
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

cap.release()
cv2.destroyAllWindows()
