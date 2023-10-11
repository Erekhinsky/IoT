from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class Centroid:
    def __init__(self, max_disappeared, max_distance):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        return object_id

    def update(self, rectangulars):
        score = []
        if len(rectangulars) == 0:

            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    score.append(self.deregister(object_id))

            return self.objects, score

        input_centroids = np.zeros((len(rectangulars), 2), dtype="int")

        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rectangulars):
            x = int((start_x + end_x) / 2.0)
            y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (x, y)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            matrix = dist.cdist(np.array(object_centroids), input_centroids)

            rows = matrix.min(axis=1).argsort()
            cols = matrix.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if matrix[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, matrix.shape[0])).difference(used_rows)
            unused_cols = set(range(0, matrix.shape[1])).difference(used_cols)

            if matrix.shape[0] >= matrix.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        score.append(self.deregister(object_id))
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects, score
