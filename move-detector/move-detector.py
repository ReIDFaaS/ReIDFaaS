import datetime
import os
import threading
import time

import cv2
import numpy as np
import requests
from flask import Flask, request, make_response

from common.BaseService import BaseService
from common.Logger import LOGGER
from common.Get_Param import get_parameters
from optical_flow import LukasKanade

app = Flask(__name__)
url_fe = get_parameters('url_fe', "http://feature-extractor.default.192.168.10.13.sslip.io/")
url_va = get_parameters('url_va', "http://target-detector.default.192.168.10.13.sslip.io/")


def connect_to_camera(stream_address):
    """
    Connects to the video source camera.
    """
    camera = None
    while camera is None or not camera.isOpened():
        try:
            camera = cv2.VideoCapture(stream_address)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        except Exception as ex:
            LOGGER.error(f'Unable to open video source: [{ex}]')
        time.sleep(1)

    return camera


class MoveDetection(BaseService):

    def __init__(
            self,
            consumer_topics=None,
            producer_topics=None,
            models=None,
            timeout=10,
            asynchronous=True):

        self.prev_frame = np.empty(0)
        self.fps = float(get_parameters('fps', 1))
        self.video_id = get_parameters('video_id', 0)
        self.video_address = get_parameters('video_address', "")

        self.optical_flow = LukasKanade()

        if models is None:
            models = ["move-detection"]
        if consumer_topics is None:
            consumer_topics = []
        if producer_topics is None:
            producer_topics = [f"move_detection_{self.video_id}"]
        self.heartbeat = time.time()
        self.connect_to_target_detector = False
        self.is_thread_running = False
        threading.Thread(target=self.process_video_from_stream, daemon=True).start()
        super().__init__(
            consumer_topics,
            producer_topics,
            models=models,
            timeout=timeout,
            asynchronous=asynchronous)
        print("Creating MoveDetection Bootstrapper module\n")

    def process_data(self, ai, data, **kwargs):
        for data_item in data:
            LOGGER.info(f"{data_item[2]} poped.")
            # 不是第一帧的情况
            if self.prev_frame.size:
                # 计算光流
                if self.optical_flow(self.prev_frame, data_item[0]):
                    LOGGER.info("Movement detected")
                    self.prev_frame = data_item[0]
                    self.producer.write_result(data)
                    self.heartbeat = time.time()
                    if self.connect_to_target_detector is False and self.is_thread_running is False:
                        response_fe = requests.get(url_fe)
                        LOGGER.error(response_fe.text)
                        self.is_thread_running = True
                        threading.Thread(target=self.invoke_video_analytics).start()
            else:
                self.prev_frame = data_item[0]
                self.heartbeat = time.time()

    def process_video_from_stream(self, timeout=20):
        """
        Processes a video accessible through a stream.
        """
        protocol = self.video_address.split(":")[0]
        LOGGER.info(f"Detected video source protocol {protocol} for video source {self.video_address}.")
        n_frame = 0
        last_snapshot = time.time()

        camera = connect_to_camera(self.video_address)

        while camera.isOpened():
            grabbed = camera.grab()

            if grabbed:
                if (time.time() - last_snapshot) >= 1 / self.fps:
                    LOGGER.info(f"Current frame index is {n_frame}.")
                    ret, frame = camera.retrieve()

                    if ret:
                        cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                        det_time = datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
                        data = (frame, det_time, n_frame)
                        self.put(data)
                        last_snapshot = time.time()
                n_frame += 1

            elif (time.time() - last_snapshot) >= timeout:
                LOGGER.debug(f"Timeout reached, releasing video source.")
                camera.release()

    def invoke_video_analytics(self):
        try:
            data = {
                'fps': self.fps,
                'video_id': self.video_id,
                'video_address': self.video_address,
                'MODEL_NAME': "detection-pedestrians-yolox"
            }

            max_retries = 100  # 最大重试次数
            for attempt in range(max_retries):
                if not self.connect_to_target_detector:
                    self.connect_to_target_detector = True
                    response = requests.post(url_va, json=data)
                    self.connect_to_target_detector = False

                    if response.status_code == 400:
                        LOGGER.warning(f"Attempt {attempt + 1}: Received status code 400. Retrying...")
                        time.sleep(10)  # 等待10秒后重试
                        continue  # 状态码为400，继续重试
                    else:
                        # 输出响应内容
                        LOGGER.error(response.text)
                        if time.time() - self.heartbeat >= 40.0:
                            LOGGER.error(f"Video analytics finished")
                            break  # 如果状态码不是400，跳出重试循环
                        else:
                            LOGGER.error(f"Video analytics not finished, retrying...")
                            time.sleep(10)  # 等待10秒后重试
                            continue  # 应当继续处理
                else:
                    break  # 如果已经连接到目标检测器，则跳出重试循环
        except Exception as ex:
            LOGGER.error(f"Error while invoking video analytics: {ex}")
        finally:
            self.is_thread_running = False


@app.route('/', methods=['POST'])
def receive_data():
    data = request.get_json()
    fps = data["fps"]
    video_id = data["video_id"]
    video_address = data['video_address']
    response = make_response({
        "msg": "Data received successfully!"
    })
    os.environ['fps'] = fps
    os.environ['video_id'] = video_id
    os.environ['video_address'] = video_address
    threading.Thread(target=MoveDetection).start()
    time.sleep(10)
    return response


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3001)
