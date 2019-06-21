#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import c3d_model

segment = 32


def readFile():
    f = open("./predict/event_32.txt", 'r')
    lines = list(f)
    f.close()

    return lines


def readTrainData(batch, line, batch_size):

    skip_frame = 0
    first_frame = 0
    events = []
    segments = []

    for b in range(batch * batch_size, batch * batch_size + batch_size):

        frames = []
        video = line[b].split(' ')[0]
        start = int(line[1][b])
        cap = cv2.VideoCapture("../../../dataset/video/" + video)

        if segment == 32:
            skip_frame = 1
            first_frame = start
        elif segment == 64:
            first_frame = random.randint(start, start + 31)
            skip_frame = 1

        for n in range(32):

            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + skip_frame*n)  # 设置要获取的帧号
            a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            b = cv2.resize(b, (c3d_model.width, c3d_model.height), interpolation=cv2.INTER_CUBIC)
            b = per_image_standard(b, c3d_model.width * c3d_model.height)
            frames.append(b)

        events.append(frames)
        segments.append([video, str(start), str(start+segment)])
    events = np.array(events).astype(np.float32)
    print(segments[0])
    return events, segments


def per_image_standard(image, num_rgb):
    mean = np.mean(image)
    stddev = np.std(image)
    image = (image - mean) / (max(stddev, 1.0 / np.sqrt(num_rgb)))
    return image
