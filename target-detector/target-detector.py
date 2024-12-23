import os
import time
from typing import List

import torch
from flask import Flask, request, make_response

from common.Logger import LOGGER
from common.Get_Param import get_parameters
from common.BaseService import BaseService

from model.bytetracker import ByteTracker

app = Flask(__name__)


class VideoAnalytics(BaseService):
    def __init__(
            self,
            consumer_topics=[],
            producer_topics=["object_detection"],
            models: List = [],
            timeout=10,
            asynchronous=True
    ):
        super().__init__(
            consumer_topics,
            producer_topics,
            models=models,
            timeout=timeout,
            asynchronous=asynchronous)
        self.heartbeat = time.time()

    def process_data(self, ai, data, **kwargs):
        result = ai.predict(data)
        if result:
            for d in result:
                self.producer.write_result(d)
        return

    def preprocess(self, data):
        self.heartbeat = time.time()
        return data

    def close(self):
        LOGGER.debug("Perform housekeeping operations.")
        self.consumer.consumer.close()
        self.producer.producer.close()


LOGGER.info(f"Loading checkpoint from disk.")
model_path = "/home/data/model/yolox.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(model_path, map_location=device)

class Bootstrapper:
    def __init__(self) -> None:
        print("Creating Detection/Tracking Bootstrapper module\n")

        self.fps = float(get_parameters('fps', 25))
        self.video_id = get_parameters('video_id', 0)
        self.video_address = get_parameters('video_address', "")

        self.service = None

    def run(self):
        """
        Entry point for the component. It decides how to process the video
        source based on its kind.
        """

        model = ByteTracker(video_id=self.video_id, checkpoint=checkpoint, device=device)
        self.service = VideoAnalytics(consumer_topics=[f"move_detection_{self.video_id}"], models=[model],
                                      asynchronous=True)

        self.close()

    def close(self, timeout=60):
        """
        Exits the worker.
        """
        while (time.time() - self.service.heartbeat) <= timeout:
            # response_va = requests.get("http://127.0.0.1:8012/")
            # # response_fe = requests.get("http://127.0.0.1:6001/")
            # LOGGER.error(response_va.text)
            time.sleep(5)

        # perform cleanup of the service
        self.service.close()
        del self.service
        LOGGER.error(f"VideoAnalysis job completed.")


@app.route('/', methods=['POST'])
def receive_data():
    data = request.get_json()
    fps = data["fps"]
    video_id = data["video_id"]
    video_address = data['video_address']
    response = make_response({
        "msg": "Data received successfully!"
    })
    os.environ['fps'] = str(fps)
    os.environ['video_id'] = str(video_id)
    os.environ['video_address'] = str(video_address)
    main_worker()
    return response

def main_worker():
    bs = Bootstrapper()
    bs.run()
    del bs

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=4001)
