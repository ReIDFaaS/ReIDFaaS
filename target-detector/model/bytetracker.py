import io
import os
import time
import torch
import numpy as np
import cv2

from common.DetTrackResult import DetTrackResult
from common.Logger import LOGGER
from common.Get_Param import get_parameters

# YOLOX imports
from yolox.yolox.exp import get_exp
from yolox.yolox.utils import pre_process, fuse_model, postprocess

# Tracking imports
from yolox.yolox.tracker.byte_tracker import BYTETracker

os.environ['BACKEND_TYPE'] = 'TORCH'


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ByteTracker:

    def __init__(self, **kwargs) -> None:
        """
        Initializes the YOLOX pedestrian detection/tracking model with the default parameters (e.g., minimum detection
        confidence, NMS threshold) to be used during inference.
        """
        # Detection parameters
        self.exp = None
        self.num_classes = get_parameters('num_classes', 1)
        self.confidence_thr = float(get_parameters('confidence_thr', 0.7))
        self.nms_thr = float(get_parameters('nms_thr', 0.45))
        self.frame_rate = int(float(get_parameters('fps', 30)))
        self.image_size = int(get_parameters('input_shape', 640))
        self.op_mode = get_parameters('op_mode', 'covid19')

        self.set_to_eval = True
        self.model = None
        self.camera_code = kwargs.get('video_id', 0)

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.swap = (2, 0, 1)

        self.device = kwargs.get('device', 'cuda')
        self.checkpoint = kwargs.get('checkpoint', None)

        # Tracking parameters
        self.tracker_args = Namespace(
            track_thresh=float(get_parameters('track_thresh', 0.5)),
            mot20=False,
            track_buffer=int(get_parameters('track_buffer', 100)),
            match_thresh=0.8,
            min_box_area=int(get_parameters('min_box_area', 200)),
            frame_rate=self.frame_rate
        )
        self.tracker = BYTETracker(args=self.tracker_args, frame_rate=self.frame_rate)

        self.model_name = "yolox_model"
        LOGGER.info(f"Loading model {self.model_name}")
        self.load()

        LOGGER.info(f"Evaluating model {self.model_name}")
        self.evaluate()

    def load(self) -> None:
        """
        Load the pre-trained weights.
        """
        if not self.model:
            # Current experiment settings
            self.exp = get_exp(os.path.join("exps/", "yolox_s_mix_det.py"), exp_name="")

            # Initialize YOLOX model
            self.model = self.exp.get_model()
            self.model.load_state_dict(self.checkpoint["model"])

            self.model = fuse_model(self.model)
            self.model.to(self.device)

    def evaluate(self, **kwargs):
        self.model.eval()

    def forward(self, data):
        """ Perform forward pass on with the provided nerual network.
            'data' is a List of images.
        """

        img_info = {}

        # Store image information, will be used during post processing
        height, width = data[0].shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        LOGGER.debug("Batching images")
        num_imgs = len(data)

        # Image info now stores the raw images
        img_info["raw_img"] = []

        # early exit
        if data is None or data[0] is None:
            return None, None

        # Get new size for forward pass
        imgp, new_size = pre_process(data[0], (self.image_size, self.image_size), self.rgb_means, self.std, self.swap,
                                     self.device)
        img_info["test_size"] = new_size

        # input_batched - create placeholder 
        if self.device == "cuda":
            # initialized directly on GPU
            input_batched = torch.cuda.FloatTensor(num_imgs, 3, new_size[1], new_size[0])
        else:
            input_batched = torch.zeros(num_imgs, 3, new_size[1], new_size[0])
        input_batched[0, :, :, :] = imgp

        # Iterate over all images and concat
        for i, img in enumerate(data):

            if i == 0:
                continue

            input_batched[i, :, :, :], _ = pre_process(data[0], (self.image_size, self.image_size), self.rgb_means,
                                                       self.std, self.swap, self.device)

        if self.device == "cpu":
            input_batched = input_batched.float()

        # Forward pass 
        with torch.no_grad():
            # Image forward pass, output now contains
            # outputs is a [batch_size, num_box, 5 + num_classes] tensor
            # outputs[:, :, 5: 5 + num_classes] contains the confidence score for each class

            # in batching mode, outputs will be of shape [b, num_boxes, 5 + num_classes] instead of [1, num_boxes,
            # 5 + num_classes]
            outputs = self.model(input_batched)

        outputs = outputs.cpu()

        outputs = postprocess(
            outputs,
            self.num_classes,
            self.confidence_thr,
            self.nms_thr)

        # Re-scale the bboxes to be consistent with the "true" image size.
        if (outputs is not None) and (outputs[0] is not None):
            outputs = outputs[0]
            bboxes = outputs[:, 0:4]
            ratio = [img_info["test_size"][0] / img_info["width"], img_info["test_size"][1] / img_info["height"]]
            ratio = np.tile(ratio, (bboxes.shape[0], 2))
            # bboxes = bboxes.cpu()
            bboxes /= ratio
            outputs[:, 0:4] = bboxes
            img_info["test_size"] = (height, width)
            outputs = [outputs]

        return outputs, img_info

    def detection(self, data, output, img_info, det_time, frame_nr):
        """
        Performs object detection.
        """

        result = None

        # Prepare image with boxes overlaid
        if output is not None:
            try:
                bboxes = output[:, 0:4]
                a0 = output[:, 4]
                a1 = output[:, 5]
                # bboxes /= img_info["ratio"]

                if len(bboxes) > 0:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    scene = cv2.imencode('.jpg', data, encode_param)[1]

                    result = DetTrackResult(
                        frame_index=frame_nr,
                        scene=scene,
                        confidence=np.multiply(a0, a1),
                        detection_time=det_time,
                        camera=self.camera_code,
                        bbox_coord=bboxes,
                        tracking_ids=[-1] * len(bboxes)
                    )

                    # self.write_to_fluentd(result)
                    LOGGER.info(f"Found {len(bboxes)} objects/s in camera {self.camera_code}")

            except Exception as ex:
                LOGGER.error(f"No objects identified [{ex}].")

        return result

    def tracking(self, data, output, img_info, det_time, frame_nr):
        """
        Performs object tracking.
        """

        # initialize placeholders for the tracking data
        online_tlwhs = []
        online_ids = []
        online_scores = []
        result = None

        original_size = (data.shape[0], data.shape[1])

        try:
            # update tracker
            online_targets = self.tracker.update(output, original_size, original_size)

            for t in online_targets:
                # bounding box and tracking id
                # tlwh - top left width height
                tlwh = t.tlwh
                tid = t.track_id

                # prior about human aspect ratio
                f_vertical = tlwh[2] / tlwh[3] > 1.6

                if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not f_vertical:
                    online_tlwhs.append(tuple(map(int, tlwh)))
                    online_ids.append(int(tid))
                    online_scores.append(t.score)

            online_bboxes = [None] * len(online_tlwhs)
            for i, t in enumerate(online_tlwhs):
                x1, y1, w, h = t
                x2 = x1 + w
                y2 = y1 + h
                online_bboxes[i] = [x1, y1, x2, y2]

            if online_tlwhs:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                scene = cv2.imencode('.jpg', data, encode_param)[1]

                result = DetTrackResult(
                    frame_index=frame_nr,
                    scene=scene,
                    confidence=online_scores,
                    detection_time=det_time,
                    camera=self.camera_code,
                    bbox_coord=online_bboxes,
                    tracking_ids=online_ids
                )

                # self.write_to_fluentd(result)
                LOGGER.info(
                    f"Tracked {len(online_bboxes)} objects/s in camera {self.camera_code} with IDs {result.tracklets}")

        except Exception as ex:
            LOGGER.error(f"No objects tracked! [{ex}].")

        return result

    def covid19(self, data, outputs, img_info, det_time, frame_nr):
        return self.tracking(data, outputs, img_info, det_time, frame_nr)

    def predict(self, data, **kwargs):
        """
        Main prediction function.
        """
        tresult = []

        start = time.time()
        imgs_ = list(map(lambda z: z[0], data))

        outputs, img_info = self.forward(imgs_)

        if len(outputs) > 0:
            for idx, output in enumerate(outputs):
                dettrack_obj = None
                try:
                    dettrack_obj = getattr(self, self.op_mode)(data[idx][0], output, img_info, data[idx][1],
                                                                     data[idx][2])
                    if dettrack_obj is not None:
                        tresult.append(dettrack_obj)
                except AttributeError as ex:
                    LOGGER.error(f"Operational mode {self.op_mode} not supported. [{ex}].")
                    return []

            LOGGER.info(f"Transmitting {len(tresult)} DetTrack objects to the next component.")

        LOGGER.info(f"~FPS:{(1 * len(imgs_)) / ((time.time() - start))}")

        return tresult
