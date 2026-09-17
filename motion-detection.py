import os
for _flag in ("QWEN_2_5_ENABLED",
    "QWEN_3_ENABLED",
    "CORE_MODEL_SAM_ENABLED",
    "CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_SAM3_ENABLED",
    "CORE_MODEL_GAZE_ENABLED"):
        os.environ[_flag] = 'False'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
import cv2
from pathlib import Path
import supervision as sv
import csv
import numpy as np


RTSP_URL='rtsp://192.168.0.242:8554/front-door-cam'
RECORDINGS = sorted(f for f in Path('./dataset/detections').iterdir() if f.suffix == '.mp4')
recording_idx=0

LIVE = False
SAVE_DATA = False
RECORDER = False


def capture():
    global recording_idx
    if LIVE:
        return cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    else:
        c = cv2.VideoCapture('./dataset/motion/cars.MP4')
        c = cv2.VideoCapture(RECORDINGS[recording_idx])
        recording_idx+=1
        return c


cap = capture()
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not LIVE else -1
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Frame count: {frame_count:,}  |  Size: {frame_width}x{frame_height}  |  FPS: {fps}')
rec = cv2.VideoWriter('./dataset/motion/file_name.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame_width,frame_height),True)

all_detections = []

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=True) # 500,16,True

prev_frame = cap.read()[1]
# prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap = capture()
        continue

    kernel = np.ones((2,2), np.uint8)
    # frame = cv2.GaussianBlur(frame, (5,5), 0)
    frame = cv2.erode(frame, kernel)
    frame = cv2.dilate(frame,kernel,iterations=1)

    # frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN, kernel) # erode and dilate together (removes noise)
    # greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    delta = cv2.absdiff(prev_frame, frame)
    cv2.imshow('delta', delta)

    # fgmask = fgbg.apply(frame)
    # cv2.imshow('back sub', fgmask)
    print(len(delta))
    # cv2.imshow(f'frame - {recording_idx}', frame)


    if RECORDER: rec.write(frame)
    prev_frame=frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        recording_idx+=1
        cap.release()
        cv2.destroyAllWindows()
        cap=capture()
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# print(all_detections)

if SAVE_DATA is True:
    with open('./dataset/detections.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['x1','y1','x2','y2','confidence','class_name','tracker_id','recording_name','frame_time','frame_num','fps', 'model_id'])
           # writer.writerow('xyxy','confidence','class_name','tracker_id')
        writer.writeheader()
        writer.writerows(all_detections)
        # for row in all_detections:
        #     writer.writerow(row)
