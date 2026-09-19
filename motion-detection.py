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
import numpy as np


RTSP_URL='rtsp://192.168.0.242:8554/front-door-cam'
RECORDINGS = sorted(f for f in Path('./dataset/detections').iterdir() if f.suffix == '.mp4')
RECORDINGS.insert(0, './dataset/motion/cars.MP4')
recording_idx=0

LIVE = False
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
FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not LIVE else -1
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 1920x1080
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Frame count: {FRAME_COUNT:,}  |  Size: {FRAME_WIDTH}x{FRAME_HEIGHT}  |  FPS: {FPS}')
rec = cv2.VideoWriter('./dataset/motion/file_name.mp4',cv2.VideoWriter_fourcc(*'mp4v'),FPS,(FRAME_WIDTH,FRAME_HEIGHT),True)



def merge_boxes(rects, grow=1):
    grow /= 100
    grow += 1 # grow 10%
    #* boundingRec: x,y,w,h (top left corner, width, height)
    #* cv2 (0,0) coord is also top left

    boxes = []
    objects = []

    rects.sort(key=lambda r: (r[0]**2 + r[1]**2))
    # for x,y,w,h in rects: #grow boxes
    #     # if w < 2 and h < 2: continue # filter out single pixel boxes

    #     # size = int((w*h)**.5) # increase based on box size
    #     # pad = int(size * grow)
    #     # x = max(0, x-pad)
    #     # y = max(0, y-pad)
    #     # w = min(w+pad, FRAME_WIDTH-x)
    #     # h = min(h+pad, FRAME_HEIGHT-y)

    #     boxes.append((x,y,w,h))
    #     # d = np.ceil(np.sqrt((w-x)**2 + (h-y)**2).astype(int) / 2)
    #     center=(x+w//2, y+h//2)
    #     centers.append(center)

    # Z = np.float32([(x,y) for x,y,_,_ in boxes])
    if len(rects):
        # items = list(zip(centers, boxes))
        items = [((x+w//2, y+h//2), (x,y,w,h)) for x,y,w,h in rects] #center and box
        centers = [c for c,_ in items]
        avg_std = int(np.mean(np.std(centers,axis=0))) # (x,y)

        if avg_std < 50: #* single detection area
            objects.append(items[0][1]) #* normalize min box size (width/height)
        else:
            # print(f'avg_std: {avg_std}')
            # std_dev = np.std(centers,axis=0)
            # mean = np.mean(centers,axis=0)
            # z_scores = (centers-mean) / std_dev
            # z_norm = np.linalg.norm(z_scores, axis=1)
            # dists = [int(np.linalg.norm(np.asarray(c) - np.asarray(c1))) for c in centers]

            while len(items) > 1:
                # c1 = items[len(centers)//2][0] #* may change to [0] over len, maybe mean?
                c1 = items[0][0] #! seed needs to be changed, hmmmmm maybe item with std_dev of 0?

                group1,group2 = [],[]
                for c,b in items:
                    dist = np.linalg.norm(np.asarray(c) - np.asarray(c1))
                    (group1 if dist < 150 else group2).append((c,b))
                if group1:
                    objects.append(group1[0][1]) #* only 1 box per group
                items = group2
            if items: #leftover after loop
                objects.append(items[0][1])
    # print(len(objects))
    return objects


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
    thresh[FRAME_HEIGHT - 70 :, FRAME_WIDTH - 550 :] = 0 # black out timer

    # thresh = cv2.adaptiveThreshold(delta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 9)
    # merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
    # thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, merge_kernel) #* too slow (with merge_kernel)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL(boxes)/RETR_TREE(all points)


    #* cv2.drawContours(orig_frame, contours, -1, (0, 255, 0), 2)
    #* contourArea is used to filter out some white noise from thresh and camera
    rectangles = [list(cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) > 20] #!20

    # grouped_rects, weights = cv2.groupRectangles(rectangles, groupThreshold=2, eps=6)

    boxes = merge_boxes(rectangles)
    # if boxes: print(len(boxes),boxes)

    for (x, y, w, h) in boxes:
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow('thresh', thresh)
    cv2.imshow('original', orig_frame)

    prev_frame=frame

    if RECORDER: rec.write(orig_frame)
    key = cv2.waitKey(max(1, int(1000/FPS))) & 0xFF
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
