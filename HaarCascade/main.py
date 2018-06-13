import numpy as np
import cv2
import math
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot
from cv2 import *

vid = cv2.VideoCapture('../video/SaltCity2.mp4', cv2.IMREAD_GRAYSCALE)  # (1)
# vid = VideoCapture(0)  # (2) 0 -> index of camera
_, frame_rgb = vid.read()
frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
avg = np.float32(frame)
print(avg)

#cascadePath = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_upperbody.xml"
cascadePath = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_fullbody.xml"
# cascadePath = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalcatface.xml"
#cascadePath = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_eye.xml"
detector = cv2.CascadeClassifier(cascadePath)
WhiteImage = np.zeros(frame.shape, frame.dtype)


nframe = 0
while(1):
    nframe = nframe+1
    _, frame_rgb = vid.read()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(frame, avg, 0.1)
    # print(avg)
    #cv2.equalizeHist(avg, avg)
    background = cv2.convertScaleAbs(avg)
    diff = cv2.absdiff(frame, background)

    cv2.imshow("input", diff)
    cv2.imshow("Background", background)
    rects = detector.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=0, minSize=(60, 60), maxSize=(160, 160))

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y,  w+x,  h+y] for (x, y, w, h) in rects])
    rects_ms = non_max_suppression(rects, probs=None, overlapThresh=0.75)

    # calculate center of detection box:
    cogs = np.array([[math.floor((x+w)/2),
                      math.floor((y+h)/2),
                      1] for (x, y, w, h) in rects_ms])

    # draw rectangles and points into image
    for rect in rects_ms:
        # draw rectangle (x,y),(w+x,h+y)
        cv2.rectangle(frame_rgb, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)
    for cog in cogs:
        cv2.circle(frame_rgb, (cog[0], cog[1]), 5, (0, 0, 255), -1)


# Homgraphy matrix (has to be changed if camera changes)
# https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html
    rot = np.array([[0.0101240234768616,     -0.00847479565930775,        -0.974192433448834],
                    [-0.00196859030258317,     -0.000929201148491731,        -0.225308693846282],
                    [9.56043341206427e-06,      4.71595084773723e-06,       -0.0024572747705754]])
# draw drawGrid
    points = np.array([[0, 0, 1],  # raw points
                       [0, 100, 1],
                       [100, 100, 1],
                       [100, 0, 1],
                       ['nan', 'nan', 'nan'],
                       [50, 0, 1],
                       [50, 100, 1],
                       ])
    points = np.transpose(np.matmul(rot, np.transpose(points)))
    points = np.array([[x/w, y/w] for (x, y, w) in points])  # devide by w (homogene coordinates)
    cv2.polylines(frame_rgb, np.int32([points]), True, (0, 255, 255))

    # Transform points to topview:
    # invrot = np.linalg.inv(rot)
    # topcogs = np.transpose(np.matmul(invrot, np.transpose(cogs)))
    # topcogs = np.array([[x/w, y/w] for (x, y, w) in topcogs])

    # plot topview
    # matplotlib.pyplot.scatter(cogs[:, 0], cogs[:, 1])

    cv2.imshow("Upper Body", frame_rgb)

    if cv2.waitKey(1) == 27:
        break
    # matplotlib.pyplot.pause(0.05)
    # matplotlib.pyplot.clf()
    #
cv2.destroyAllWindows()
