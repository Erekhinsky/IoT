from streamlink import Streamlink
from centroid import Centroid
from imutils.video import VideoStream
from itertools import zip_longest
from imutils.video import FPS
import numpy as np
import argparse
import datetime
import imutils
import dlib
import json
import csv
import cv2

with open("config.json", "r") as file:
    config = json.load(file)


class TrackableObject:
    def __init__(self, object_id, centroid):
        self.object_id = object_id
        self.centroids = [centroid]
        self.counted = False
        self.last_direction = 0


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("-p", "--prototxt", required=False)
    arg.add_argument("-m", "--model", required=True)
    arg.add_argument("-i", "--input", type=str)
    arg.add_argument("-o", "--output", type=str)
    arg.add_argument("-c", "--confidence", type=float, default=confidence_arg)
    arg.add_argument("-s", "--skip-frames", type=int, default=skip_frames)
    args = vars(arg.parse_args())
    return args


def log(move_in, time_in, move_out, time_out):
    data = [move_in, time_in, move_out, time_out]
    export_data = zip_longest(*data, fillvalue='')

    with open('data/logs/counting_data.csv', 'w', newline='') as datafile:
        writer = csv.writer(datafile, quoting=csv.QUOTE_ALL)
        if datafile.tell() == 0:
            writer.writerow(("|Move In|", "|Time In|", "|Move Out|", "|Time Out|"))
            writer.writerows(export_data)


def stream_to_url(url, quality='best'):
    session = Streamlink()
    streams = session.streams(url)
    if streams:
        return streams[quality].to_url()


skip_count = 70
confidence_arg = 0.7
skip_frames = 20
max_disappeared = 30
max_distance = 170
mean_color = 150


def main():
    global objects, status
    args = parse_arguments()
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # if not args.get("input", False):
    #     print("Starting the live stream..")
    #     video_capture = VideoStream(config["url"]).start()
    #     time.sleep(2.0)
    # else:
    #     print("Starting the video..")
    #     video_capture = cv2.VideoCapture(args["input"])

    if config["Thread"]:
        print("Starting the live stream...")
        stream_url = stream_to_url(config["url"], '480p')
        video_capture = cv2.VideoCapture(stream_url)
    elif args.get("input", True):
        print("Starting the video...")
        video_capture = cv2.VideoCapture(args["input"])
    else:
        print("Starting the live stream...")
        video_capture = VideoStream(config["url"], cv2.CAP_FFMPEG).start()

    writer = None
    width = None
    height = None

    ct = Centroid(max_disappeared=max_disappeared, max_distance=max_distance)
    trackers = []
    trackable_objects = {}

    frames_num = 0
    in_num = 0
    out_num = 0

    total = []
    move_out = []
    move_in = []
    time_out = []
    time_in = []

    fps = FPS().start()
    skip_counter = 0

    while True:
        ret, frame = video_capture.read()

        if args["input"] is not None and frame is None:
            break

        frame = imutils.resize(frame, width=1000, height=1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if width is None or height is None:
            (height, width) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (width, height), True)

        rectangulars = []
        if frames_num % args["skip_frames"] == 0:
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.006, (width, height), mean_color)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (start_x, start_y, end_x, end_y) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rectangle = dlib.rectangle(start_x, start_y, end_x, end_y)
                    tracker.start_track(rgb, rectangle)
                    trackers.append(tracker)

        else:
            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()
                start_x = int(pos.left())
                start_y = int(pos.top())
                end_x = int(pos.right())
                end_y = int(pos.bottom())

                rectangulars.append((start_x, start_y, end_x, end_y))

        line_locate = height // 2
        cv2.line(frame, (0, line_locate), (width, line_locate), (255, 255, 255), 3)
        cv2.putText(frame, "Entrance", (20, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)

        objects, score = ct.update(rectangulars)

        for (object_id, centroid) in objects.items():
            trackable_object = trackable_objects.get(object_id, None)
            # немного другая реализация подсчета - счет не между кадрами, а по выходу из кадра
            # id_control.add(object_id)
            # for object_id_score in score:
            #     print(score)
            #     id_control.remove(object_is_score)
            #     trackable_objects_score = trackable_objects.get(object_is_score, None)
            #     if trackable_objects_score.last_direction < 0:
            #         out_num += 1
            #         date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            #         move_out.append(out_num)
            #         time_out.append(date_time)
            #         # trackable_objects_score.counted = True
            #         score.remove(object_id_score)
            #     else:
            #         in_num += 1
            #         date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            #         move_in.append(in_num)
            #         time_in.append(date_time)
            #         trackable_objects_score.counted = True
            #         score.remove(object_id_score)
            #     total = [len(move_in) - len(move_out)]

            if trackable_object is None:
                trackable_object = TrackableObject(object_id, centroid)
            else:
                # минус - выход , плюс - вход
                if skip_counter == skip_count:
                    skip_counter = 0
                    y = [cor[1] for cor in trackable_object.centroids]
                    direction = centroid[1] - np.mean(y)
                    trackable_object.centroids.append(centroid)
                    if not trackable_object.counted:
                        if direction < 0 and centroid[1] < line_locate:
                            out_num += 1
                            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                            move_out.append(out_num)
                            time_out.append(date_time)
                            trackable_object.counted = True
                            trackable_object.last_direction -= 1
                        elif direction > 0 and centroid[1] > line_locate:
                            in_num += 1
                            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                            move_in.append(in_num)
                            time_in.append(date_time)
                            trackable_object.counted = True
                            trackable_object.last_direction += 1

                        total = [in_num - out_num]
                    # else:
                    #     if direction < 0 and centroid[1] < line_locate:
                    #         trackable_objects.last_direction -= 1
                    #     elif direction > 0 and centroid[1] > line_locate:
                    #         trackable_objects.last_direction += 1
                else:
                    skip_counter += 1

            trackable_objects[object_id] = trackable_object

            text = "ID {}".format(object_id)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        status = [
            ("Exit", out_num),
            ("Enter", in_num),
            ("Total inside", total)]

        for (i, (j, k)) in enumerate(status):
            text = "{}: {}".format(j, k)
            cv2.putText(frame, text, (10, height - 30 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)

        if sum(total) >= config["Limit"]:
            cv2.putText(frame, "Number of people is too big", (width - 260, height - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 100, 255), 2)

        log(move_in, time_in, move_out, time_out)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Analysis video window", frame)
        cv2.waitKey(20)

        frames_num += 1
        fps.update()

    # for object_id in objects.keys():
    #     trackable_objects_score = trackable_objects.get(object_id, None)
    #     if trackable_objects_score.last_direction < 0:
    #         out_num += 1
    #         date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    #         move_out.append(out_num)
    #         time_out.append(date_time)
    #         trackable_objects_score.counted = True
    #     else:
    #         in_num += 1
    #         date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    #         move_in.append(in_num)
    #         time_in.append(date_time)
    #         trackable_objects_score.counted = True

    # total = [len(move_in) - len(move_out)]
    # log(move_in, time_in, move_out, time_out)

    fps.stop()

    if config["Thread"]:
        video_capture.release()

    cv2.destroyAllWindows()


main()
