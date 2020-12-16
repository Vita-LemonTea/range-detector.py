# import the necessary packages
from centroidtracker import CentroidTracker
from detector import TargetDetector
import pandas as pd
import argparse
import time
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

# initialize the centroid tracker and target detector
ct = CentroidTracker()
dt = TargetDetector()
# initialize an empty dataframe to store result
df = pd.DataFrame(columns=['ID', 'center', 'frame'])
# initialize the video
print("[INFO] starting the video")
vs = cv2.VideoCapture(args["video"])
# allow the video file to warm up
time.sleep(2.0)

# loop over the frames from the video
framecount = 1
while True:
    # read the next frame from the video
    frame = vs.read()
    frame = frame[1]

    # get bounding box rectangles from target detector
    rects = dt.detector(frame)

    # update centroid tracker using the computed set of bounding box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw the IDs and the centroids of the objects on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
        # add object location to dataframe
        df = df.append([{'ID': objectID, 'center': tuple(centroid), 'frame': framecount}])
    # draw the bounding box rectangle on the frame
    for r in rects:
        x, y, x_e, y_e = r
        cv2.rectangle(frame, (x, y), (x_e, y_e), (255, 0, 0), 2)
    # count the frame
    framecount += 1
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# write the result
df.to_csv("result.csv", encoding="gbk", index=False )
# cleanup
cv2.destroyAllWindows()
vs.release()
