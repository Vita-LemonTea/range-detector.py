# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max=50):
        # initialize the next unique object ID and two ordered
        # dictionaries of current objects and disappeared objects,respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # initialize the max number of disappeared frame
        self.maxDisappeared = max

    def update(self, rects):
        # if the list of input bounding box rectangles is empty,
        # mark existing objects as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # if object reaches max disappeared frame, deregister it
                if self.disappeared[object_id] > self.maxDisappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]

            # return early as there are no centroids or tracking info to update
            return self.objects

        # if the list of input bounding box rectangles is not empty
        # initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # calculate centroids for each bounding box rectangles
            cx = int(sum([startX, endX]) / 2.0)
            cy = int(sum([startY, endY]) / 2.0)
            input_centroids[i] = (cx, cy)

        # if currently there is no tracking object, take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.objects[self.nextObjectID] = input_centroids[i]
                self.disappeared[self.nextObjectID] = 0
                self.nextObjectID += 1

        # otherwise, there are currently tracking objects
        # match the input centroids to existing object centroids
        else:
            # initialize the set of object IDs and corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute the Euclidean distance between each pair of
            # object centroids and input centroids
            distance = dist.cdist(np.array(object_centroids), input_centroids)

            # sort the distance from small to large
            rows = distance.min(axis=1).argsort()
            cols = distance.argmin(axis=1)[rows]
            combination = zip(rows, cols)

            # initialize rows and cols already examined
            examined_rows = set()
            examined_cols = set()

            # loop over the combination of distance
            for (row, col) in combination:
                # skip the examined rows and cols
                if row in examined_rows or col in examined_cols:
                    continue

                # set new centroid as object ID for the current row
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # update examined rows and cols
                examined_rows.add(row)
                examined_cols.add(col)

            # compute both the rows and cols that not examined
            unexamined_rows = set(range(0, distance.shape[0])).difference(examined_rows)
            unexamined_cols = set(range(0, distance.shape[1])).difference(examined_cols)

            # if number of current object centroids is equal or
            # greater than the number of input centroids
            # check whether some objects disappeared
            if distance.shape[0] >= distance.shape[1]:
                # loop over the unused row indexes
                for row in unexamined_rows:
                    # mark the object as disappeared
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    # if object reaches max disappeared frame, deregister it
                    if self.disappeared[object_id] > self.maxDisappeared:
                        del self.objects[object_id]
                        del self.disappeared[object_id]
            # if number of current object centroids is smaller
            # than the number of input centroids
            else:
                for col in unexamined_cols:
                    self.objects[self.nextObjectID] = input_centroids[col]
                    self.disappeared[self.nextObjectID] = 0
                    self.nextObjectID += 1

        # return the set of trackable objects
        return self.objects
