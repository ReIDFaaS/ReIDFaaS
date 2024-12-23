import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from common.DetTrackResult import DetTrackResult
from common.Logger import LOGGER


def create_results_folder(folder):
    """Creates result folder if it doesn't exist"""
    Path(folder).mkdir(parents=False, exist_ok=True)


def write_text_on_image(img, text, textX, textY, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, _ = img.shape
    tl = round(0.002 * (height + width) * 0.1) + 1

    # add text centered on image
    cv2.putText(img, text, (textX, textY), font, 0.5, color, tl)


def draw_bbox(image, bbox, text="Target", color=(255, 0, 0), tracking_id=None):
    height, width, _ = image.shape
    tl = round(0.002 * (height + width) * 0.1) + 1  # line thickness

    target_centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
    write_text_on_image(image, text, int(bbox[0]), int(bbox[1]) - 10, color)

    if tracking_id:
        write_text_on_image(image, str(tracking_id), int(bbox[0]), int(bbox[3]) - 10, color)

    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, tl)
    cv2.circle(image, target_centroid, radius=3, color=color, thickness=tl)

    return target_centroid


def save_image(data: DetTrackResult, folder):
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)

    image_name = data.image_key.rsplit("_")[0] + ".jpg"
    image = cv2.imdecode(data.scene, cv2.IMREAD_COLOR)

    # Add camera
    write_text_on_image(image, f"Camera:{data.camera}", 0, 30)

    # TARGET
    for ids in data.matchIDs:
        # LOGGER.error(f"ids: {ids}")
        idx = ids[1]
        current_image = np.copy(image)
        draw_bbox(current_image, data.bbox_coord[idx], tracking_id=data.tracklets[idx])

        # Save result in the specified folder
        base_image = Image.fromarray(current_image).resize((640, 480))
        output_directory = os.path.join(folder, ids[0])
        os.makedirs(output_directory, exist_ok=True)  # 如果目录不存在则创建
        base_image.save(os.path.join(output_directory, image_name), format='JPEG', quality=80, optimize=True)

    return image_name

