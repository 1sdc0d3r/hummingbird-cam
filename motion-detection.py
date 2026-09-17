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
import pandas as pd


RTSP_URL='rtsp://192.168.0.242:8554/front-door-cam'
RECORDINGS = sorted(f for f in Path('./dataset/detections').iterdir() if f.suffix == '.mp4')
RECORDINGS.insert(0, './dataset/motion/cars.MP4')
recording_idx=1

LIVE = False
SAVE_DATA = False
RECORDER = False

def capture():
    global recording_idx
    if LIVE:
        return cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    else:
        c = cv2.VideoCapture(RECORDINGS[recording_idx])
        # c = cv2.VideoCapture('./dataset/motion/cars.MP4')
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

#* didn't work well
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=True) # 500,16,True

kernel = np.ones((4,4), np.uint8)
prev_frame = cap.read()[1]
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.erode(prev_frame, kernel)
prev_frame = cv2.dilate(prev_frame,kernel,iterations=1)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap = capture()
        continue

    orig_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.erode(frame, kernel)
    # frame = cv2.dilate(frame,kernel,iterations=1)
    frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN, kernel) # erode and dilate together (removes noise)

    # blur = cv2.GaussianBlur(frame, (5,5), 0)
    # bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # fgmask = fgbg.apply(frame)

    delta = cv2.absdiff(prev_frame, frame)

    _,thresh = cv2.threshold(delta, 80, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(delta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 9)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL(boxes)/RETR_TREE(all points)
    # print(contours)
    # np.savetxt(f'./contours.csv', contours, delimiter=',', fmt='%d')
    break
    #! the contours are good, but now I want to group multiple together for obj identification

    # merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, merge_kernel)

    # big = [c for c in contours if cv2.contourArea(c) > 100]
    cv2.drawContours(orig_frame, contours, -1, (0, 255, 0), 2)

    # for cnt in big:
        # pass
        # if cv2.contourArea(cnt) > 200:
        # x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(orig_frame, (x,y),(x+w,y+h), (0,255,0), 2)


    # cv2.imshow('thresh', thresh)
    cv2.imshow('original', orig_frame)

    prev_frame=frame

    if RECORDER: rec.write(frame)
    key = cv2.waitKey(max(1, int(1000/fps))) & 0xFF
    if key == ord('n'):
        recording_idx+=1
        cap.release()
        cv2.destroyAllWindows()
        cap=capture()
    # if key == ord('f'):
    #     cur_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    #     np.savetxt(f'./dataset/motion/delta/frame_delta_{cur_frame_pos}.csv', delta, delimiter=',', fmt='%d')
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
