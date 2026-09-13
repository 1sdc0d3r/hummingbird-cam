import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

from skimage import data
import cv2
import matplotlib.pyplot as plt


rtsp_url = 'rtsp://192.168.0.242:8554/front-door-cam'
# image = data.astronaut()
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    exit("Could not open RTSP stream")

ret, frame = cap.read()
# drag a rectangle on the image, press ENTER/SPACE
rx, ry, rw, rh = cv2.selectROI("pick watch area", frame, False)
cv2.destroyWindow("pick watch area")
coords=[rx, ry, rw, rh]
while True:
    # break
    ret,frame = cap.read()
    if not ret:
        print('Failed to get frame')
        break

    roi = frame[ry:ry+rh, rx:rx+rw].copy()
    blurr = cv2.GaussianBlur(frame, (5,5),0)
    edges = cv2.Canny(blurr, 100,200)

    # image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # purpose?
    # greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.resize(image, (400,600))
    # cv2.resize(greyscale, (50,70))
    # cv2.resize(edges, (400,600))
    # cv2.imshow('blurr', blurr)
    # cv2.imshow('stream img', image)
    # cv2.imshow('stream greyscale', greyscale)
    # cv2.imshow('stream edges', edges)

    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    cv2.drawContours(frame,contours, -1,(0,255,0),2)
    cv2.imshow('cont',contours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#
# image = cv2.imread('vlcsnap-01.png')
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# edges = cv2.Canny(gray, 100,200)
#
# plt.figure(figsize = (10,4))
#
# plt.subplot(1,2,1)
# plt.imshow(gray, cmap='gray')
# plt.title('Grayscale Img')
# plt.axis('off')
#
# plt.subplot(1,2,2)
# plt.imshow(edges, cmap='gray')
# plt.title('Edge Img')
# plt.axis('off')
#
# plt.show()
