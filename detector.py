from collections import deque
import numpy as np
import cv2
import imutils
import time
import joblib

# define the lower and upper boundaries of the "black" larva in the HSV color space
blackLower = (0, 0, 130)
blackUpper = (255, 255, 255)


class TargetDetector:
    def detector(self, frame):
        # load ROI config file and crop the frame
        model = joblib.load('config.pkl')
        coor = np.array(model['ROI'])

        # reshape the coor according to the points of ROI
        coor = coor.reshape((4, 2))

        mask = np.zeros((frame.shape[0], frame.shape[1]))

        cv2.fillConvexPoly(mask, coor, 1)
        mask = mask.astype(np.bool)

        out = np.zeros_like(frame)
        out[mask] = frame[mask]
        frame = out

        # resize the frame, blur it, and convert it to the HSV color space
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform a series of dilations
        # and erosions to remove any small blobs left in the mask
        mask = cv2.inRange(hsv, blackLower, blackUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        contour = []
        rect = []
        for i in range(len(cnts)):
            if cnts[i].shape[0] > 18 and cnts[i].shape[0] < 40:
                contour.append(cnts[i])
        for i in range(len(contour)):
            c = contour[i]
            x, y, w, h = cv2.boundingRect(c)
            x_e = x + w
            y_e = y + h
            r = (x, y, x_e, y_e)
            rect.append(r)
        return rect
