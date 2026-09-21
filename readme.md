todo:
- convert paths to root of project (MAKE HABIT)
- create pipeline from motion-detection -> tracker_detections (live multiple graphs??)

motion-detection.py
    - #! use vectors to predict motion rather than just double loop?
    - merge_rectangles && update_tracker need some TLC. I am getting many FP, and new id being assigned to prev objects
    - [h264 @ 0x7fd8f5995540] error while decoding MB 46 22, bytestream -5 (fix this)
    - tracker and merger need more work!
