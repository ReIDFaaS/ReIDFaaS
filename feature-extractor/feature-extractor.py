import os
import pickle
import threading
import time
import cv2
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List

from functools import reduce

from flask import Flask, request, jsonify, make_response
from common.Logger import LOGGER
from common.BaseService import BaseService, FileOperations
from common.Get_Param import get_parameters
from common.DetTrackResult import DetTrackResult, Target

os.environ['BACKEND_TYPE'] = 'TORCH'

app = Flask(__name__)


class FeatureExtraction(BaseService, FileOperations):

    def __init__(
            self,
            consumer_topics=["object_detection"],
            producer_topics=["enriched_object"],
            models: List = [],
            timeout=10,
            asynchronous=False
    ):

        super().__init__(
            consumer_topics,
            producer_topics,
            models=models,
            timeout=timeout,
            asynchronous=asynchronous)
        self.heartbeat = time.time()

    def process_data(self, ai, data, **kwargs):
        for ai in self.models:
            result = ai.inference(data)

            if result:
                for d in result:
                    self.write_to_disk(data=d, folder="/home/data/features/")

    def preprocess(self, data):
        self.heartbeat = time.time()
        return data

    def update_operational_mode(self, status):
        pass


class FeatureExtractionAI:

    def __init__(self, **kwargs):
        """
        Initialize feature extraction module
        """

        self.model = None

        # Device and input parameters
        if torch.cuda.is_available():
            self.device = "cuda"
            LOGGER.info("Using GPU")
        else:
            self.device = "cpu"
            LOGGER.info("Using CPU")

        image_size = get_parameters('input_shape', "256,128")
        self.image_size = [int(image_size.split(",")[0]),
                           int(image_size.split(",")[1])]

        # Data transformation
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model_name = "m3l_model"
        LOGGER.info(f"Loading model {self.model_name}")
        self.load()

        LOGGER.info(f"Evaluating model {self.model_name}")
        self.evaluate()

    def load(self):
        """Load the pre-trained FE weights."""
        if not self.model:
            model_path = "/home/data/model/m3l.pth"
            self.model = torch.load(model_path, map_location=torch.device(self.device))
            self.model.to(self.device)

    def evaluate(self):
        """Turn eval mode on for the model."""

        LOGGER.debug(f"Setting Feature Extraction module to eval mode.")
        self.model.eval()

    def extract_features(self, data: List[DetTrackResult]):
        """
        Extract ReID features from the provided image.
        """
        input_batch = None
        j = 0
        offset = 0

        try:
            all_bboxes = list(map(lambda x: x.bbox_coord, data))  # list of lists
            total_bboxes = reduce(lambda count, l: count + len(l), all_bboxes, 0)

            # Prepare the batch
            for idx, bbox_group in enumerate(all_bboxes):

                # Decode the original image
                imdata = cv2.imdecode(data[idx].scene, cv2.IMREAD_COLOR)

                for elem in bbox_group:
                    # Using the bbox coordinates we crop the original image
                    x0, y0, x1, y1 = int(elem[0]), int(elem[1]), int(elem[2]), int(elem[3])
                    crop = Image.fromarray(imdata[y0: y1, x0: x1])

                    LOGGER.info(f'Performing feature extraction for received image.')
                    input = self.transform(crop)

                    if j == 0:
                        if self.device == "cuda":
                            # initialized directly on GPU
                            input_batch = torch.cuda.FloatTensor(total_bboxes, input.shape[0], input.shape[1],
                                                                 input.shape[2])
                        else:
                            input_batch = torch.zeros(total_bboxes, input.shape[0], input.shape[1], input.shape[2],
                                                      dtype=torch.float)

                    input_batch[j, :, :, :] = input.to(self.device)
                    j += 1

            # do forward pass once
            with torch.no_grad():
                query_feat = self.model(input_batch)
                qf = query_feat.to(self.device)

            qf = qf.to('cpu')

            # Enrich DetTrackResult object with the extracted features
            for k, bbox_group in enumerate(all_bboxes):
                num_person = len(bbox_group)
                for j in range(offset, num_person + offset):
                    f = torch.unsqueeze(qf[j, :], 0)
                    # np.expand_dims(qf[i, :], 0)
                    data[k].features.append(f)
                offset += num_person

            LOGGER.info(f"Extracted features for {offset} object/s.")

        except Exception as ex:
            LOGGER.error(f"Unable to extract features [{ex}].")
            return None

        return data

    def update_plugin(self, update_object, **kwargs):
        pass

    def get_target_features(self, ldata):
        """Extract the features for the query image. This function is invoked when a new query image is provided."""

        try:
            results = []
            for new_query_info in ldata:
                LOGGER.info(
                    f"Received {len(new_query_info.bbox)} sample images for the target for user {new_query_info.userID}.")
                new_query_info.features = []

                for image in new_query_info.bbox:
                    # new_query_info contains the query image.
                    try:
                        query_img = Image.fromarray(image[:, :, :3])  # dropping non-color channels
                    except Exception as ex:
                        LOGGER.error(f"Query image not found. Error [{ex}].")
                        return None

                    # Attempt forward pass
                    try:
                        input = torch.unsqueeze(self.transform(query_img), 0).to(self.device)
                        with torch.no_grad():
                            query_feat = self.model(input)
                            LOGGER.debug(f"Extracted tensor with features: {query_feat}.")

                        query_feat = query_feat.to('cpu')

                        # It returns a tensor, it should be transformed into a list before TX
                        new_query_info.features.append(query_feat)
                        new_query_info.is_target = True

                    except Exception as ex:
                        LOGGER.error(f"Feature extraction failed for query image. Error [{ex}].")
                        return None

                results.append(Target(new_query_info.userID, new_query_info.features))

            return results

        except Exception as ex:
            LOGGER.error(f"Unable to extract features for the target [{ex}].")

        return None

    def inference(self, data, **kwargs):
        return self.predict(data, **kwargs)

    def predict(self, data, **kwargs):
        """Implements enrichment of DetTrack object with features for ReID."""

        dettrack_objs_with_features = self.extract_features(data)

        if dettrack_objs_with_features:
            return dettrack_objs_with_features

        return []


model = FeatureExtractionAI()


class Bootstrapper:
    def __init__(self):
        self.service = FeatureExtraction(producer_topics=[], models=[model], asynchronous=True)

    def run(self, timeout=30) -> None:
        while (time.time() - self.service.heartbeat) <= timeout:
            response_fe = requests.get("http://127.0.0.1:8012/")
            LOGGER.error(response_fe.text)
            time.sleep(15)

        LOGGER.info(f"feature extraction service completed.")


call_num = 0


@app.route('/', methods=['GET', 'POST'])
def my_api():
    global call_num
    call_num += 1
    if request.method == 'GET':
        # 处理GET请求
        data = {"message": "GET请求成功"}
    elif request.method == 'POST':
        # 处理POST请求
        data = request.get_json()
        data["message"] = "POST请求成功"
    else:
        data = {"message": "无效的请求"}
    if call_num == 1:
        threading.Thread(target=main_worker).start()
    return jsonify(data)

@app.route('/receive_target', methods=['POST'])
def receive_target():
    data = request.data
    ldata = pickle.loads(data)
    features = model.get_target_features(ldata)
    features_bytes = pickle.dumps(features)
    response = make_response(features_bytes)
    return response


def main_worker():
    bs = Bootstrapper()
    bs.run()


# Starting the FE service.
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6001)
