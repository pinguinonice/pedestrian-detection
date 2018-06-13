# based on: https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import math
import cv2
from cv2 import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import time
fig = plt.figure(figsize=(2, 2))
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# video (1) or webcam (2)
# 2015 Salt City 5k on surveillance camera
# https://www.youtube.com/watch?v=HKGGAe6k2O4
cam = cv2.VideoCapture('../video/SaltCity2.mp4')  # (1)

# cam = VideoCapture(0)  # (2) 0 -> index of camera
begin = time.time()
# loop over the frames
print("A  B  C  D   T  ")
while True:
    # Start time
    start = time.time()

    # load frame
    s, frame_rgb = cam.read()
    # resize:
    # frame_rgb = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame_rgb, winStride=(4, 8),
                                            padding=(0, 0), scale=1.05)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    rects_ms = non_max_suppression(rects, probs=None, overlapThresh=0.9)
    # calculate center of detection box:
    cogs = np.array([[math.floor((x+w)/2),
                      math.floor((y+h/2)),  # red point : fits quite good
                      1] for (x, y, w, h) in rects_ms])

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in rects_ms:
        cv2.rectangle(frame_rgb, (xA, yA), (xB, yB), (0, 255, 0), 2)
    for cog in cogs:
        cv2.circle(frame_rgb, (cog[0], cog[1]), 5, (0, 0, 255), -1)


# Homgraphy matrix (has to be changed if camera changes)
# https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html
    rot = np.array([[0.00840278273967597, - 0.000996762883651406, - 0.972939274817017],
                    [-0.000611229109482202, - 0.000363960872335956, - 0.230899966154323],
                    [4.75921782278034e-06,      6.0886306800187e-06, - 0.00150556507364792]])

    # controls position in topview
    view = np.array([[-1, 0, 100],
                     [0, 1, 0],
                     [0, 0, 1]])

    # define gridpoints in top Koordinates
    points_org = np.array([[0, 0, 1],  # raw points
                           [0, 100, 1],
                           [50, 100, 1],
                           [50, 0, 1],
                           [100, 0, 1],
                           [100, 60, 1],
                           [0, 60, 1],
                           [0, 0, 1],
                           [100, 0, 1],
                           [100, 100, 1],
                           [0, 100, 1],
                           ])
    # Transpose points to image and draw them
    points = np.transpose(np.matmul(rot, np.transpose(points_org)))
    points = np.array([[x/w, y/w] for (x, y, w) in points])  # devide by w (homogene coordinates)
    cv2.polylines(frame_rgb, np.int32([points]), False, (0, 255, 255))

    # Transform  Detection points to topview:
    invrot = np.matmul(view, np.linalg.inv(rot))

    if not cogs.any():
        cogs = np.array([[0, 0, 1]])

    topcogs = np.transpose(np.matmul(invrot, np.transpose(cogs)))
    topcogs = np.array([[x/w, y/w] for (x, y, w) in topcogs])

    # count points in polygon
    A = 0  # set to zero for every frame
    B = 0
    C = 0
    D = 0
    areaA = Polygon([(0, 0), (50, 0), (50, 60), (0, 60)])  # define poligons of area
    areaB = Polygon([(50, 0), (100, 0), (100, 60), (50, 60)])
    areaC = Polygon([(0, 60), (50, 60), (50, 100), (0, 100)])
    areaD = Polygon([(50, 60), (100, 60), (100, 100), (50, 100)])
    for topcog in topcogs:
        if areaA.contains(Point(topcog)):
            A += 1
        if areaB.contains(Point(topcog)):
            B += 1
        if areaC.contains(Point(topcog)):
            C += 1
        if areaD.contains(Point(topcog)):
            D += 1
    print(("{}  {}  {}  {}  {}".format(A, B, C, D, np.round(start-begin, 2))))
    # plot topvie

    cv2.polylines(frame_rgb, np.int32([points_org[:, :2]]), False, (0, 255, 255))
    for topcog in topcogs:
        cv2.circle(frame_rgb, (np.int32(topcog[0]),
                               np.int32(topcog[1])), 2, (0, 0, 255), -1)
    cv2.putText(frame_rgb, str(A), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_rgb, str(B), (65, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_rgb, str(C), (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_rgb, str(D), (65, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    cv2.putText(frame_rgb, "FPS:{}".format(np.round(1/seconds, 1)), (20, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # show some information on the number of bounding boxes
    cv2.putText(frame_rgb, ("Number of People:{}".format(len(rects))),
                (20, 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
                )
    # plt.scatter(topcogs[:, 0], topcogs[:, 1])
    # plt.axis('equal')
    # plt.plot(points_org[:, 0], points_org[:, 1], 'y-')
    # plt.axis([-50, 150, -50, 150])
    # plt.pause(0.05)
    # plt.clf()
    # show the output images
    # cv2.imshow("Before NMS", orig)

    cv2.imshow("Detection", frame_rgb)
    # esc to quit
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cam.release()
