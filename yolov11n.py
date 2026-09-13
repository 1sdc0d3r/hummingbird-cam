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
from inference import get_model
import supervision as sv
import csv


API_KEY = os.environ.get('ROBOFLOW_API_KEY')
RTSP_URL='rtsp://192.168.0.242:8554/front-door-cam'
# MODEL_ID = 'braden-gr0sv/hummingbirds-p7kwn-1-yolo11n-t3' # v3
MODEL_ID = 'braden-gr0sv/hummingbirds-p7kwn-2-yolo11n-t1' # v4 - overfit
# MODEL_ID = 'braden-gr0sv/hummingbirds-p7kwn-2-yolo11n-t2' # v5 overfit
RECORDINGS = sorted(f for f in Path('./dataset/detections').iterdir() if f.suffix == '.mp4')
recording_idx=0

LIVE = False
SAVE_DATA = True


model = get_model(model_id=MODEL_ID, api_key=API_KEY)
tracker = sv.ByteTrack(
    lost_track_buffer=60,
    track_activation_threshold=.7,
    minimum_matching_threshold=.7,
    minimum_consecutive_frames=2,
    frame_rate=30
)


def capture():
    global recording_idx
    if LIVE:
        return cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    else:
        c = cv2.VideoCapture(RECORDINGS[recording_idx])
        recording_idx+=1
        return c

cap = capture()

all_detections = []
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap = capture()
        continue

    results = model.infer(frame)
    detections = sv.Detections.from_inference(results[0])
    detections = tracker.update_with_detections(detections)
    filtered_detections = detections[detections.confidence >= .8]
    # print(f'detections: {filtered_detections}')

    # print(detections)
    for xyxy, confidence,  class_id, tracker_id in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
        det = {
            #  'xyxy': list([str(x) for x in xyxy]),
             'x1':int(xyxy[0]),
             'y1':int(xyxy[1]),
             'x2':int(xyxy[2]),
             'y2':int(xyxy[3]),
             'confidence': round(float(confidence),3),
             'class_name': str(detections.data.get('class_name')[class_id]),
             'tracker_id': int(tracker_id),
             'recording_name': RECORDINGS[recording_idx].name,
             'frame_time': int(cap.get(cv2.CAP_PROP_POS_MSEC)),
             'frame_num':int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
             'fps': int(cv2.CAP_PROP_FPS)
        }
        all_detections.append(det)
        print(det)

    if detections.tracker_id is None:
        labels = [f'{confidence:.2f}' for class_name, confidence in zip(filtered_detections['class_name'], filtered_detections.confidence)]
    else:
        labels = [f'{tracker_id} - {confidence:.2f}' for tracker_id, confidence in zip(filtered_detections.tracker_id, filtered_detections.confidence)]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    out_img = box_annotator.annotate(scene=frame, detections=filtered_detections)
    out_img = label_annotator.annotate(scene=out_img, detections=filtered_detections, labels=labels)

    cv2.imshow(f'Hummingbird Detector - {recording_idx}', out_img)


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
        writer = csv.DictWriter(f, fieldnames=['x1','y1','x2','y2','confidence','class_name','tracker_id','recording_name','frame_time','frame_num','fps'])
           # writer.writerow('xyxy','confidence','class_name','tracker_id')
        writer.writeheader()
        writer.writerows(all_detections)
        # for row in all_detections:
        #     writer.writerow(row)
