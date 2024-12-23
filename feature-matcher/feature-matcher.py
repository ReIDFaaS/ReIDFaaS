import io
import os
import pickle
import time

import requests
import torch
import numpy as np
from PIL import Image
from threading import Thread
from typing import List

from flask import Flask, request, make_response
from common.BaseService import BaseService, FileOperations

from store_result import save_image
from multi_img_matching import match_query_to_targets
from common.Logger import LOGGER
from common.DetTrackResult import DetTrackResult, Target
from common.Get_Param import get_parameters

url_fe = get_parameters('url_fe', "http://feature-extractor.default.192.168.10.82.sslip.io/")
app = Flask(__name__)


class ReID(BaseService, FileOperations):

    def __init__(
            self,
            consumer_topics=[],
            producer_topics=[],
            models: List = [],
            timeout=10,
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

    def __init__(self, **kwargs):
        # Service parameters
        self.op_mode = get_parameters('op_mode', 'covid19')
        self.threshold = get_parameters('match_threshold', 0.70)
        self.user_id = get_parameters('user_id', "DEFAULT")
        LOGGER.info("user id: %s", self.user_id)

        self.target = None
        self.targets_list: List[Target] = []

    def update_plugin(self, status):
        # Update target
        LOGGER.info("Loading target query images")

        # The target collection is a list of targets/userid that might grow overtime
        img_arr = []

        for file in os.listdir(f"/home/data/query/{self.user_id}"):
            LOGGER.info(f"Loading file {file}...")
            image_bytes = open(os.path.join(f"/home/data/query/{self.user_id}/", file), "rb").read()
            img_arr.append(np.asarray(Image.open(io.BytesIO(image_bytes))))

        data = DetTrackResult(0, img_arr, None, [], 0, 0)
        data.userID = self.user_id

        return [data]

    def update_target(self, ldata):
        """
            Updates the target for the ReID.       
        """
        LOGGER.info(f"Target updated for user {ldata[0].userid} with {len(ldata[0].features)} feature vectors!")
        self.targets_list = ldata

    def reid_per_frame(self, features, candidate_feats: torch.Tensor) -> int:
        """
        For each frame, this function receives the ReID features for all the detected boxes. The similarity is computed
        between the query features and the candidate features (from the boxes). If matching score for all detected boxes
        is less than match_thresh, the function returns None signifying that no match has been found. Else, the function
        returns the index of the candidate feature with the highest matching score.
        @param candidate_feats: ...
        @return: match_id [int] which points to the index of the matched detection.
        """

        if features == None:
            LOGGER.warning("Target has not been set!")
            return -1

        match_id, match_score = match_query_to_targets(features, candidate_feats, False)
        return match_id, match_score

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

        for target in self.targets_list:
            # get id of highest match for each userid
            match_id, match_score = self.reid_per_frame(target.features, det_track.features)

            result = {
                "userID": str(target.userid),
                "match_score": str(match_score),
                "detection_area": det_track.camera,
                "detection_time": det_track.detection_time
            }
            # print(self.threshold)
            if float(match_score) >= self.threshold:
                det_track.targetID[match_id] = str(target.userid)
                det_track.userID = target.userid
                det_track.is_target = match_id
                LOGGER.info(result)

            # LOGGER.info(result)
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
            save_image(det_track, folder=f"/home/data/images/{det_track.userID}/")

        except Exception as ex:
            LOGGER.error(f"Unable to save image: {ex}")


class Bootstrapper(Thread):
    def __init__(self):
        super().__init__()

        self.daemon = True
        self.retry = 3
        self.job = ReID(models=[ReIDWorker()], asynchronous=True)

    def run(self) -> None:
        LOGGER.info("Loading data from local storage.")
        # Load data from local storage
        LOGGER.info(f"Loading files ...")
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


@app.route('/', methods=['POST'])
def receive_data():
    start_time = time.time()
    data = request.get_json()
    user_id = data["user_id"]
    response = make_response({
        "start_time": start_time
    })
    os.environ['user_id'] = user_id
    bs = Bootstrapper()
    bs.run()
    return response


# Start the ReID job.
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
