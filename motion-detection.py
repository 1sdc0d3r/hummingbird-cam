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
from uuid import uuid4
from datetime import datetime as dt

#todo convert to full numpy for speed
#? keep consistent w/h sizes? only use center or x/y?

RTSP_URL='rtsp://192.168.0.242:8554/front-door-cam'
RECORDINGS = sorted(f for f in Path('./dataset/detections').iterdir() if f.suffix == '.mp4')
RECORDINGS.insert(0, './dataset/motion/cars.MP4')
recording_idx=10

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

def get_rec_center(x,y,w,h):
    return (x+w//2, y+h//2)

def merge_contours(rects=[], grow=1):
    #* contourArea is used to filter out some white noise from thresh and camera
    rects = [list(cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) > 20] #!20
    # grow /= 100
    # grow += 1 # grow 10%
    #* boundingRec: x,y,w,h (top left corner, width, height)
    #* cv2 (0,0) coord is also top left

    # boxes = []
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
                #* maybe use a seed from the prev frame??

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

# live_tracker = [{'uuid':'uuid()','TTL':15,'center':(0,0),'count':0, 'test':True}] #* frame nums?
live_tracker=[]
def update_tracker(new, tracker=live_tracker):
    centers = [(x+w//2, y+h//2) for x,y,w,h in new]

    #! use vectors to predict motion rather than just double loop
    for center in centers:
        center = np.array(center)

        trk = dict()
        dist = -1
        for t in tracker:
            t_center = np.array(t.get('center')) #convert all code later
            l2 = int(np.linalg.norm(center-t_center))
            print('l2',l2)
            if l2 < dist or dist == -1:
                dist=l2
                trk = t

        if trk and dist < 50:
            trk['TTL'] = 60
            trk['count'] += 1
            trk['center'] = center
            # centers.remove(center)
        else:
            #* new trackers
            new_obj = {'uuid':uuid4(),'TTL':60,'center':center,'count':1}
            tracker.append(new_obj)
            print(f'new-{center}')


    #* remove expired trackers (save to db)
    for t in tracker:
        if t['TTL'] <= 0:
            print(f"remove-{t['center']}")
            tracker.remove(t)




'''
    for t in tracker: #! don't do fcfs. check dist on every tracker first. swap centers and trackers. separate logic
        t['TTL'] -= 1
        for i, new_center in enumerate(centers.copy()):
            t_center = np.array(t.get('center')) #convert all code later
            new_center = np.array(new_center)
            l2 = int(np.linalg.norm(new_center-t_center))

            # print(f't_center:{t_center} - new_center:{new_center} - l2: {l2}')
            if l2 < 50:
                t['TTL'] = 60
                t['count'] += 1
                t['center'] = new_center
                centers.pop(i)

        #* remove old trackers (save to db)
        if t['TTL'] <= 0:
            print(f"remove-{t['center']}")
            tracker.remove(t)

    #* new trackers
    for c in centers:
        new_obj = {'uuid':uuid4(),'TTL':60,'center':c,'count':1}
        tracker.append(new_obj)
        print(f'new-{c}')

'''

#* didn't work well
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=True) # 500,16,True

kernel = np.ones((4,4), np.uint8)
prev_frame = cap.read()[1] #
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.morphologyEx(prev_frame, cv2.MORPH_OPEN, kernel)
prev_objects = []

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap = capture()
        continue

    cur_frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if cur_frame_count % 10 == 0:
        #  print(cur_frame_count)
        for t in live_tracker:
            print(t)
        print('-'*5)

        #  break
    # if cur_frame_count > 500:
    #     print(live_tracker)
    #     break

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
    # grouped_rects, weights = cv2.groupRectangles(rectangles, groupThreshold=2, eps=6)

    objects = merge_contours(contours)

    update_tracker(objects)

    for (x, y, w, h) in objects:
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # if prev_objects:
        # cv2.line(orig_frame, (x,y), (prev_objects[0][0], prev_objects[0][1]), (255,0,0),2)

    # cv2.imshow('thresh', thresh)
    cv2.imshow(f'original - {recording_idx}', orig_frame)


    prev_frame = frame
    prev_objects = objects # for tracking when obj stops

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
