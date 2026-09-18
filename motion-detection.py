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


def merge_boxes(rects, grow=10):
    grow /= 100
    grow += 1 # grow 10%
    #* boundingRec: x,y,w,h (top left corner, width, height)
    #* cv2 (0,0) coord is also top left

    boxes = []
    for x,y,w,h in rects: #grow boxes
        size = (w+h)/4 # increase based on box size
        pad = int(size * grow)
        x = max(0, x-pad)
        y = max(0, y-pad)
        w = min(w+2*pad, frame_width-x)
        h = min(h+2*pad, frame_height-y)

        if w > 1 and h > 1: # filter out single pixel boxes
            boxes.append((x,y,w,h)) 


    # print(f'{rects}\n{boxes}')
    return boxes

    # 1920x1080


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
    frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN, kernel) # erode and dilate together (removes noise)


    delta = cv2.absdiff(prev_frame, frame)

    _,thresh = cv2.threshold(delta, 80, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(delta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 9)

    # merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
    # thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel) #* too slow (with merge_kernel)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL(boxes)/RETR_TREE(all points)

    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, merge_kernel)

    #* cv2.drawContours(orig_frame, contours, -1, (0, 255, 0), 2)
    rectangles = [list(cv2.boundingRect(c)) for c in contours]
    # rectangles += rectangles  # Ensure proper weighting for groupThreshold=1
    # rectangles = merge_boxes(rectangles)

    # grouped_rects, weights = cv2.groupRectangles(rectangles, groupThreshold=2, eps=6)
    # print(len(rectangles), len(grouped_rects), weights)
    # print(len(rectangles))
    boxes = merge_boxes(rectangles)
    print(len(boxes),boxes)

    for (x, y, w, h) in boxes:
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)


    # big = [c for c in contours if cv2.contourArea(c) > 30]
    # print(len(big))
    # for c in contours:
    #     x,y,w,h = cv2.boundingRect(c)
    #     cv2.rectangle(orig_frame, (x,y),(x+w,y+h), (0,255,0), 2)
    #     continue
        # if cv2.contourArea(c) > 50: #300


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


def merge_rects(rects, pad=20):
    boxes = [[x - pad, y - pad, x + w + pad, y + h + pad] for x, y, w, h in rects]
    merged = True
    while merged:
        merged = False
        out = []
        while boxes:
            a = boxes.pop()
            ax1, ay1, ax2, ay2 = a
            rest = []
            for b in boxes:
                bx1, by1, bx2, by2 = b
                if ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1:
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    merged = True
                else:
                    rest.append(b)
            boxes = rest
            out.append([ax1, ay1, ax2, ay2])
        boxes = out
    return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
