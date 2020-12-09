# import the necessary packages
from centroidTracker import CentroidTracker
from detector import TargetDetector
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
                help="max buffer size")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
dt = TargetDetector()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
# allow the video file to warm up
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # read the next frame from the video stream and resize it
    frame = vs.read()
    frame = frame[1]
    #frame = imutils.resize(frame, width=600)

    rects = dt.detector(frame)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
