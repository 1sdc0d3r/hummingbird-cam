import sys
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
from matplotlib import pyplot as plt

plt.ion()
fig, ax = plt.subplots()
graph_hist = {}

#todo convert to full numpy for speed
#? keep consistent w/h sizes? only use center or x/y?

RTSP_URL='rtsp://192.168.0.242:8554/front-door-cam'
RECORDINGS = sorted(f for f in Path('./dataset/detections').iterdir() if f.suffix in ('.mp4','.MP4','.mov'))
RECORDINGS[:0] = [f for f in Path('./dataset/motion').iterdir() if f.suffix in ('.mp4','.MP4','.mov')]
# RECORDINGS.extend(sorted(f for f in Path('./dataset/motion').iterdir() if f.suffix in ('.mp4','.MP4','.mov')))
# RECORDINGS.insert(0, './dataset/motion/cars.MP4')
# RECORDINGS.insert(0, './dataset/motion/truck1.MP4')
FEEDER = (120,650,280,130) # [120:400, 650:780]
FEEDER2 = (75,200,325,125) #[75:275, 325:450]

LIVE = False
RECORDER = False
LIVE_GRAPH = False
SKIP_FRAMES=False
recording_idx=20 #10


if not LIVE_GRAPH: plt.close('all')

def capture():
    global recording_idx,graph_hist
    if LIVE:
        return cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    else:
        cv2.destroyAllWindows()
        c = cv2.VideoCapture(RECORDINGS[recording_idx])
        # c = cv2.VideoCapture('./dataset/motion/cars.MP4')
        recording_idx+=1
        graph_hist = {}
        return c

cap = capture()
FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not LIVE else -1
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 1920x1080
FPS = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Frame count: {FRAME_COUNT:,}  |  Size: {FRAME_WIDTH}x{FRAME_HEIGHT}  |  FPS: {FPS}')
rec_path = f'./dataset/vectors/{dt.now()}'
rec = cv2.VideoWriter(f'{rec_path}/recording.mp4',cv2.VideoWriter_fourcc(*'mp4v'),FPS,(FRAME_WIDTH,FRAME_HEIGHT),True)

def get_rec_center(x,y,w,h):
    return (x+w//2, y+h//2)

def grow_rectangles(rects, grow=1):
    grow /= 100
    grow += 1 # grow 10%
    boxes = []
    for x,y,w,h in rects: #grow boxes
            # if w < 2 and h < 2: continue # filter out single pixel boxes
            size = int((w*h)**.5) # increase based on box size
            pad = int(size * grow)
            x = max(0, x-pad)
            y = max(0, y-pad)
            w = min(w+pad, FRAME_WIDTH-x)
            h = min(h+pad, FRAME_HEIGHT-y)

            boxes.append((x,y,w,h))
            # d = np.ceil(np.sqrt((w-x)**2 + (h-y)**2).astype(int) / 2)
            # center=(x+w//2, y+h//2)
            # centers.append(center)

        # Z = np.float32([(x,y) for x,y,_,_ in boxes])
    return boxes

def by_feeder(center):
    x, y, w, h = FEEDER
    x1, y1, w1, h1 = FEEDER2

    cx, cy = center
    return (x <= cx < x + w and y <= cy < y + h) or (x1 <= cx < x1 + w1 and y1 <= cy < y1 + h1)

#* normalize min box size (width/height)
def set_min_box(item,m=25):
    x = list(item)
    if x[2] < m: x[2]=m
    if x[3] < m: x[3]=m
    return x

def get_max_box(items):
    boxes = [b for _,b in items]
    x=min(b[0] for b in boxes)
    y=min(b[1] for b in boxes)
    x2=max(b[0]+b[2] for b in boxes)
    y2=max(b[1]+b[3] for b in boxes)
    return set_min_box((x,y,x2-x,y2-y))

def get_box_size(item):
    x,y,w,h = item
    return (w,h)

#* PRINT ALL RECTANGLES - mini
def draw_objects(rectangles):
    for (x,y,w,h) in rectangles:
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 0), 1)
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

#* PRINT OBJ RECTANGLES
def draw_live_tracker(tracker, orig_frame, thresh):
    for t in tracker:
        if t['count'] < 4: continue #* filters out some noisy trackers - run ttl down
        x, y, w, h = t['box']
        # w,h = t['box_size_avg']

        #*frame,text,pos,font,fontScale,color,lineType
        cv2.putText(orig_frame, str(t['uuid'])[-1:-4:-1], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,50,255), 2)
        cv2.putText(thresh, str(t['uuid'])[-1:-4:-1], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,50,255), 2)
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 0), 2)

def merge_rectangles(rects):
    #* boundingRec: x,y,w,h (top left corner, width, height)
    #* cv2 (0,0) coord is also top left
    objects = []
    rects.sort(key=lambda r: (r[0]**2 + r[1]**2))

    if len(rects):
        # items = list(zip(centers, boxes))
        items = [((x+w//2, y+h//2), (x,y,w,h)) for x,y,w,h in rects] #center and box
        centers = [c for c,_ in items]
        avg_std = np.mean(np.std(centers,axis=0)) # (x,y)
        # print(items,avg_std)
        # print(centers, np.std(centers, ddof=1))
        if avg_std < 50: #* single detection area # 50
            center_mean = np.mean(centers, axis=0)
            i = np.argmin(np.linalg.norm(centers-center_mean, axis=1))
            obj = get_max_box(items)
            # obj = set_min_box(items[i][1])
            objects.append(obj)
        else:
            # print('avg',avg_std)
            # print(f'avg_std: {avg_std}')
            # std_dev = np.std(centers,axis=0)
            # mean = np.mean(centers,axis=0)
            # z_scores = (centers-mean) / std_devqq
            # z_norm = np.linalg.norm(z_scores, axis=1)
            # dists = [int(np.linalg.norm(np.asarray(c) - np.asarray(c1))) for c in centers]

            # print('*'*5)
            while len(items) > 1:
                centers = [c for c,_ in items]
                # print(centers,avg_std)
                # print(centers, np.mean(np.std(centers, ddof=0,axis=0)))
                # center_mean = np.mean(np.linalg.norm(centers, axis=1))
                center_mean = np.mean(centers, axis=0)
                i = np.argmin(np.linalg.norm(centers-center_mean, axis=1))
                seed = items[i]

                group1,group2 = [],[]
                for c,b in items:
                    dist = np.linalg.norm(np.asarray(c) - np.asarray(seed[0]))
                    (group1 if dist < 60 else group2).append((c,b)) #! 150,80(good)
                    #! set dist value based on group size? std and size? (truck and trailer)
                # objects.append(get_max_box(group1))
                objects.append(set_min_box(seed[1])) #* only 1 box per group (c1 seed)

                items = group2
            if items: #leftover after loop
                objects.append(set_min_box(items[0][1]))
    return objects


live_tracker=[]
def update_tracker(objects, frame_num, tracker=live_tracker):
    #* remove expired trackers (save to db)
    for t in tracker:
        t['TTL'] -= 1
        if t['TTL'] <= 0:
            tracker.remove(t)

    centers = [(x+w//2, y+h//2) for x,y,w,h in objects]

    #! find the mean size of object for consistent size (truck behind pillar, ect)
    #! do I care about speed across skipped frames? last_frame-cur_frame in trk so velocity per frame?
    #! clear out noisy trackers
    #! track accuracy (report on q) FP count, comp of alg used
    #! boxes still up after off camera (fix)
    for i,center in enumerate(centers):
        center = np.array(center)
        ttl = FPS*5 if by_feeder(center) else FPS//2 #* birds sitting or hovering so dont use pred alg
        trk = dict()
        score = -1
        for t in tracker:
            #* use predictive tracker on non-feeder areas
            t_center = np.array(t['center'])
            s = np.linalg.norm(center-t_center) if by_feeder(center) else np.linalg.norm(center-t['prediction'])
            if s < score or score == -1:
                trk = t
                score = s

        # err_center,err_pred = int(dist),int(low_pred)
        # diff = err_center-err_pred
        # print(trk is t2, err_center, err_pred, diff)

        if trk and score < 200: #! reduce val as acc inc
            trk['TTL'] = ttl
            trk['count'] += 1
            trk['center'] = center
            trk['box'] = objects[i]
            trk['last_frame'] = frame_num
            trk['trace'].append(center)
            trk['box_sizes'].append(get_box_size(objects[i])) #! this is wrong <-- get_box_size. this is where you start next... store w,h separate (not just area)
            trk['box_size_avg'] = np.mean(trk['box_sizes'], axis=0).astype(int).tolist()
            trk['by_feeder'] = by_feeder(center)

            #* vector math
            a = trk['init_center'] # growing magnitude as leaves origin point
            a = trk['trace'][-2] # prev
            b = center # current
            velocity = b-a
            magnitude = np.linalg.norm(velocity)
            trk['velocity'] = velocity
            trk['prediction'] = center + velocity
            # print(f"{str(trk['uuid'])[-4:]} <{velocity},{magnitude}>")
            #? direction? L/R for cars? extract later from data or..
            if LIVE_GRAPH:
                # each frame when you print a vector
                tid = str(trk['uuid'])[-3:]
                graph_hist.setdefault(tid, []).append(magnitude)
                ax.cla()
                for k, v in graph_hist.items():
                    ax.plot(v, 'o-', label=k, markersize=3)
                ax.legend(loc='best', fontsize=8)
                ax.set_ylabel('magnitude')
                plt.pause(0.001)

        else:
            #* new trackers
            new_obj = {
                'uuid':uuid4(),
                'TTL':ttl,
                'center':center,
                'init_center':center,
                'box':objects[i],
                'count':1,
                'init_frame':frame_num,
                'last_frame':frame_num,
                'trace':[center],
                'box_sizes': [get_box_size(objects[i])],
                'box_size_avg': get_box_size(objects[i]),#* changes to numpy later, db issue? needed,hmm
                'prediction': center,
                'by_feeder': by_feeder(center), #nice to have for the db
                }
            tracker.append(new_obj)
            # print(f'new-{center}-{dist}')


#* didn't work well
fgbg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=120, detectShadows=False) # 500,16,True

kernel = np.ones((3,3), np.uint8) #* ODD (1 in None)

prev_frame = cap.read()[1]
# prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# prev_frame = cv2.morphologyEx(prev_frame, cv2.MORPH_OPEN, kernel)
# prev_frame = cv2.morphologyEx(prev_frame, cv2.MORPH_CLOSE, kernel)

if SKIP_FRAMES: cap.set(cv2.CAP_PROP_POS_FRAMES, 160)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        cap = capture()
        continue

    cur_frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if cur_frame_count % 10 == 0:
        # print('-'*5)
        # print('frame:',cur_frame_count)
        for t in live_tracker:
            # print(t)
            pass

    orig_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cur_frame_count < 4: # stops recordings from delta changes
        prev_frame = frame
        continue

    # frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN, kernel) # erode and dilate together (removes noise)
    # frame = cv2.morphologyEx(frame,cv2.MORPH_CLOSE, kernel)
    delta = cv2.absdiff(prev_frame, frame)
    _,thresh = cv2.threshold(delta,55, 255, cv2.THRESH_BINARY) #! 5,60-lower causes more noise
    # thresh = fgbg.apply(frame, learningRate=-1)
    thresh[FRAME_HEIGHT - 70 :, FRAME_WIDTH - 550 :] = 0 # black out timer
    # orig_frame[120:400, 650:780] = 0 #FEEDER

    # thresh = cv2.adaptiveThreshold(delta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 9)
    # merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
    # thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, merge_kernel) #* too slow (with merge_kernel)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL(boxes)/RETR_TREE(all points)


    #* cv2.drawContours(orig_frame, contours, -1, (0, 255, 0), 2)
    # grouped_rects, weights = cv2.groupRectangles(rectangles, groupThreshold=2, eps=6)

    #* contourArea is used to filter out some white noise from thresh and camera
    rectangles = [list(cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) > 20] #! 20


    draw_objects(rectangles)
    objects = merge_rectangles(rectangles)
    update_tracker(objects,cur_frame_count)
    draw_live_tracker(live_tracker, orig_frame, thresh)

#! IMSHOW
    # cv2.imshow(f'thresh - {recording_idx}', thresh)
    cv2.imshow(f'original - {recording_idx}', orig_frame)

    # if cur_frame_count % 2:
    prev_frame = frame
    if RECORDER: rec.write(orig_frame) #record thresh too

    key = cv2.waitKey(max(1, int(1000/FPS))) & 0xFF
    if key == ord('n'):
        # recording_idx+=1
        cap.release()
        cap = capture()
        prev_frame = cap.read()[1]
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        graph_hist.clear()
        live_tracker.clear()
        # fgbg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=120, detectShadows=False)

    # if key == ord('f'):
    #     cur_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    #     np.savetxt(f'./dataset/motion/delta/frame_delta_{cur_frame_pos}.csv', delta, delimiter=',', fmt='%d')
    if key == ord('q'):
        if RECORDER: plt.savefig(f'{rec_path}/graph.png') #? on next?
        break

cap.release()
cv2.destroyAllWindows()
plt.close('all')
