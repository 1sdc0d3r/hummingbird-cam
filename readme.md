todo:
- convert paths to root of project (MAKE HABIT)
- create pipeline from motion-detection -> tracker_detections (live multiple graphs??)
- trim tracker obj down. make smaller (cardinality)? whatever is best for speed
- setup parallel processing? (wyze-bridge is backing up causing streaming errors)

motion-detection.py
    - #! use vectors to predict motion rather than just double loop?
    - merge_rectangles && update_tracker need some TLC. I am getting many FP, and new id being assigned to prev objects
    - [h264 @ 0x7fd8f5995540] error while decoding MB 46 22, bytestream -5 (fix this with parallel processing?)
    - tracker and merger need more work!

    - tracker: uuid for recording videos.
    - use model for classification once (twice validation?) per object id after count > 3.(run model separately?)

database:
    what to do for testing/tuning? repeat data with different results??
    sessions table
    frame num on live? use timestamp
    Tables: trackers,traces(append-only),box_sizes(append-only), sessions
        STORES VECTORS??
        tracker - trim down object? have center col be a generated column?, camera_id, source, mode (video/live), count
        traces - x,y,w,h (cx,cy calculated cols)

    - drop points 'under pressure' where bird hovers without moving


cleanup between videos/capture
