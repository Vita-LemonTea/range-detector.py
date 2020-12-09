from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import joblib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())


# define the lower and upper boundaries of the "black"
# larva in the HSV color space
blackLower = (0, 0, 150)
blackUpper = (255, 255, 255)

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

vs = cv2.VideoCapture(args["video"])
# allow the video file to warm up
time.sleep(2.0)


# keep looping
while True:
	# grab the current frame
	frame = vs.read()
	frame = frame[1]
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# load ROI config file and crop the frame
	model = joblib.load('config.pkl')
	coor = np.array(model['ROI'])
	coor = coor.reshape((8, 2))

	mask = np.zeros((frame.shape[0], frame.shape[1]))

	cv2.fillConvexPoly(mask, coor, 1)
	mask = mask.astype(np.bool)

	out = np.zeros_like(frame)
	out[mask] = frame[mask]
	frame = out

	# resize the frame, blur it, and convert it to the HSV color space
	frame = imutils.resize(frame, width=600)
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
	for i in range(len(cnts)):
		if cnts[i].shape[0] > 10 and cnts[i].shape[0] < 20:
			contour.append(cnts[i])
	# only proceed if at least one contour was found
	if len(contour) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		#c = max(cnts, key=cv2.contourArea)
		c = contour[0]
		#((x, y), radius) = cv2.minEnclosingCircle(c)
		x, y, w, h = cv2.boundingRect(c)
		x_e = x + w
		y_e = y + h
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if w*h > 1:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			pts.appendleft(center)

		# loop over the set of tracked points
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
		# show the frame to our screen
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
		vs.stop()
# otherwise, release the camera
else:
		vs.release()
# close all windows
cv2.destroyAllWindows()


