import io
import os
import pickle
import random
import string
import threading
import time

import requests
import torch
import numpy as np
from PIL import Image
from threading import Thread
from typing import List, Any, Tuple

from flask import Flask, request, make_response
from common.BaseService import BaseService, FileOperations

from store_result import save_image
from common.Logger import LOGGER
from common.DetTrackResult import DetTrackResult, Target
from common.Get_Param import get_parameters

url_fe = get_parameters('url_fe', "http://feature-extractor.default.192.168.10.13.sslip.io/")
app = Flask(__name__)


def tensor_reshape(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        if len(data.shape) > 2:
            data = data.squeeze(0)

    if isinstance(data, List):
        if len(data[0].shape) > 2:
            temp = [x.squeeze(0) for x in data]
            data = torch.cat(temp, dim=0)
        else:
            data = torch.cat(data, dim=0)

    return data


def match_query_to_targets(query_feats: List,
                           candidate_feats: List,
                           avg_mode: bool = False) -> Tuple[int, float]:


    # candidate_feats = tensor_reshape(candidate_feats).to(device)
    if avg_mode:
        # average query_feats
        query_feats = torch.mean(query_feats, dim=0).unsqueeze(0)

    # compare features
    sim_dist = torch.mm(query_feats, candidate_feats.t())

    max_val, idx = torch.max(sim_dist, dim=1)
    global_max_val, max_index = torch.max(max_val, dim=0)
    match_id = idx[max_index].item()

    return match_id, global_max_val.item()

class ReID(BaseService, FileOperations):

    def __init__(
            self,
            consumer_topics=[],
            producer_topics=[],
            models: List = [],
            timeout=0.01,
            asynchronous=True
    ):

        self.models = models
        self.heartbeat = time.time()
        super().__init__(
            consumer_topics,
            producer_topics,
            models=models,
            timeout=timeout,
            asynchronous=asynchronous)

    def _post_init(self):
        super()._post_init()
        self.update_operational_mode(None)

    def process_data(self, ai, data, **kwargs):
        self.heartbeat = time.time()
        ai.predict(data)
        return

    def update_operational_mode(self, status):
        for ai in self.models:
            try:
                ldata = ai.update_plugin(status)
                target_list = self.get_target_features(ldata)
                ai.update_target(target_list)
            except Exception as ex:
                LOGGER.error(
                    f"Unable to update AI parameters/configuration for service {ai.__class__.__name__}. [{ex}]")
                LOGGER.error(
                    f"The exception was raised in: {ex.__traceback__.tb_frame.f_code.co_filename}, line {ex.__traceback__.tb_lineno}")
        return

    def get_target_features(self, ldata):
        ldata_bytes = pickle.dumps(ldata)
        url = url_fe + '/receive_target'
        data = ldata_bytes
        response = requests.post(url, data=data)
        features_bytes = response.content
        features = pickle.loads(features_bytes)
        return features


class ReIDWorker:

    def __init__(self, query_targets = [], **kwargs):
        # Service parameters
        self.op_mode = get_parameters('op_mode', 'covid19')
        self.threshold = get_parameters('match_threshold', 0.70)
        # self.user_id = get_parameters('user_id', "DEFAULT")
        self.query_targets = query_targets
        for target in self.query_targets:
            LOGGER.info("user id: %s", target)

        self.target = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.targets_list: List[Target] = []

    def update_plugin(self, status):
        # Update target
        LOGGER.info("Loading target query images")

        data_list = []
        for target in self.query_targets:
            img_arr = []

            for file in os.listdir(f"/home/data/query/{target}"):
                LOGGER.info(f"Loading file {file}...")
                image_bytes = open(os.path.join(f"/home/data/query/{target}/", file), "rb").read()
                img_arr.append(np.asarray(Image.open(io.BytesIO(image_bytes))))

            data = DetTrackResult(0, img_arr, None, [], 0, 0)
            data.userID = target
            data_list.append(data)

        return data_list

    def update_target(self, ldata):
        """
            Updates the target for the ReID.       
        """
        for target in ldata:
            LOGGER.info(f"Target updated for user {target.userid} with {target.features} feature vectors!")
            target.features = tensor_reshape(target.features).to(self.device)
            self.targets_list.append(target)

    def predict(self, data, **kwargs):
        """Implements the on-the-fly ReID where detections per frame are matched with the candidate boxes."""
        tresult = []

        for dettrack_obj in data:
            try:
                reid_result = getattr(self, self.op_mode + "_no_gallery")(dettrack_obj)
                if reid_result is not None:
                    tresult.append(reid_result)
                    self.store_result(reid_result)
            except AttributeError as ex:
                LOGGER.error(f"Error in dynamic function mapping. [{ex}]")
                return tresult

        return tresult

    ### OP_MODE FUNCTIONS ###

    def covid19_no_gallery(self, det_track):
        return self.tracking_no_gallery(det_track)

    def detection_no_gallery(self, det_track):
        LOGGER.warning(f"This operational mode ({self.op_mode}) is not supported without gallery.")
        return None

    def tracking_no_gallery(self, det_track: DetTrackResult):
        """
            Performs ReID without gallery using the results from the 
            tracking and feature extraction component.       
        """
        det_track.targetID = [-1] * len(det_track.bbox_coord)
        det_track.features = tensor_reshape(det_track.features).to(self.device)
        for target in self.targets_list:
            # get id of highest match for each userid
            match_id, match_score = match_query_to_targets(target.features, det_track.features, False)

            result = {
                "userID": str(target.userid),
                "match_score": str(match_score),
                "detection_area": det_track.camera,
                "detection_time": det_track.detection_time
            }
            # print(self.threshold)
            if float(match_score) >= self.threshold:
                det_track.targetID[match_id] = str(target.userid)
                # det_track.userID = target.userid
                det_track.matchIDs = []
                det_track.matchIDs.append([str(target.userid), match_id])
                det_track.is_target = True
                # LOGGER.error(f"Match found for user {target.userid} with score {match_score}")

                LOGGER.info(result)
        # print(det_track.targetID)
        if det_track.targetID.count(-1) == len(det_track.targetID):
            # No target found, we don't send any result back
            return None

        return det_track

    def store_result(self, det_track: DetTrackResult):
        """
            Stores ReID result on disk (and OBS, if enabled).     
        """
        try:
            save_image(det_track, folder=f"/home/data/images/")

        except Exception as ex:
            LOGGER.error(f"Unable to save image: {ex}")


class Bootstrapper(Thread):
    def __init__(self, current_requests=[]):
        super().__init__()

        self.daemon = True
        self.retry = 3
        self.job = ReID(models=[ReIDWorker(current_requests)], asynchronous=True)

    def run(self) -> None:
        LOGGER.info("Loading data from local storage.")

        for file in os.listdir("/home/data/features/"):
            if file.endswith(".dat"):
                # LOGGER.info(f"Loading file {file}...")
                data = self.job.read_from_disk(os.path.join("/home/data/features/", file))
                if data:
                    # LOGGER.info(f"File {file} loaded!")
                    self.job.put(data)
        LOGGER.info(f"Files loaded!")
        while len(self.job.sync_queue) > 0:
            time.sleep(0.1)
        self.job.stop_thread = True
        LOGGER.error("ReID job completed.")
        LOGGER.info("ReID job completed.")


# 用于存储请求数据
request_queue = []
request_lock = threading.Lock()
# 定义处理请求的时间间隔
process_interval = 1  # 一秒处理一次
results = {}  # 用于存储每个用户ID的处理结果
batch_size = int(get_parameters('batch_size', 5))  # 每次处理的请求数量
heartbeat = time.time()  # 用于记录服务的心跳时间

def process_requests():
    global request_queue, results, heartbeat
    while True:
        if len(request_queue) >= batch_size or (time.time() - heartbeat > process_interval and len(request_queue) > 0):
            with request_lock:
                # 处理当前请求队列的副本
                current_requests = request_queue[:5]
                request_queue = request_queue[5:]
            # 处理请求
            user_ids = [request[0] for request in current_requests]
            bs = Bootstrapper(user_ids)
            bs.run()
            for request in current_requests:
                results[request[1]] = True  # 标记处理完毕
            heartbeat = time.time()  # 更新心跳时间
        else:
            time.sleep(0.01)  # 等待请求队列中的请求

@app.route('/', methods=['POST'])
def receive_data():
    start_time = time.time()
    global request_queue, results
    data = request.get_json()
    user_id = data["user_id"]
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))


    # 将请求添加到队列
    with request_lock:
        request_queue.append([user_id, random_suffix])

    # 等待处理结果的生成
    while not random_suffix in results:
        time.sleep(0.01)  # 等待处理结果生成

    del results[random_suffix]  # 清除结果
    response = make_response({
        "start_time": start_time
    })
    return response


# Start the ReID job.
if __name__ == '__main__':
    # 启动处理线程
    threading.Thread(target=process_requests, daemon=True).start()
    app.run(debug=False, host='0.0.0.0', port=5001)
