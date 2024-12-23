import numpy
import cv2
from common.Logger import LOGGER

__all__ = 'LukasKanade'


class LukasKanade:

    """
        Class to detect movement between two consecutive images.
    """
    def __init__(self, **kwargs):
        # Parameters for ShiTomas corner detection
        self.feature_params = \
            dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Parameters for Lucas Kanade optical flow
        _criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(_criteria, 10, 0.03),
        )

    def __call__(self, old_frame=None, current_frame=None):
        """
        :param old_frame: prev_image to compare against
        :param current_frame: next_image to check for motion
        :return: `True` means that there is movement in two subsequent frames,
             `False` means that there is no movement.
        """

        movement = False
        try:
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(
                old_gray, mask=None, **self.feature_params)

            current_gray = \
                cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, current_gray, p0, None, **self.lk_params
            )

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # We perform rounding because there might ba a minimal difference
            # even between two images of the same subject
            # (image compared against itself)
            # Allclose is used instead of array_equal to support
            # array of floats (if we remove rounding).
            movement = \
                not numpy.allclose(
                    numpy.rint(good_new),
                    numpy.rint(good_old)
                )
        except Exception as ex:
            LOGGER.error(
                f"Error during the execution of\
                     the optical flow estimation! [{ex}]")

        return movement


class LukasKanadeCUDA:

    """
        Class to detect movement between
        two consecutive images (GPU implementation).
    """
    def __init__(self, **kwargs):
        # Parameters for ShiTomasi corner detection
        self.feature_params = \
            dict(
                srcType=cv2.CV_8UC1,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7)

        # Parameters for Lucas Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2
        )

        self.corner_detector = \
            cv2.cuda.createGoodFeaturesToTrackDetector(
                **self.feature_params
            )
        self.of = \
            cv2.cuda.SparsePyrLKOpticalFlow_create(**self.lk_params)

    def __call__(self, old_frame=None, current_frame=None):
        """
        :param old_img: prev_image to compare against
        :param current_img: next_image to check for motion
        :return: `True` means that there is movement in two subsequent frames,
             `False` means that there is no movement.
        """

        old_frame = cv2.cuda_GpuMat(old_frame)
        current_frame = cv2.cuda_GpuMat(current_frame)

        movement = False
        try:
            old_gray = \
                cv2.cuda.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = self.corner_detector.detect(old_gray)

            current_gray = \
                cv2.cuda.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            p1, st, err = self.of.calc(
                old_gray, current_gray, p0, None
            )

            # Select good points
            p0 = p0.download().astype(numpy.float32)
            p1 = p1.download().astype(numpy.float32)
            st = st.download().astype(numpy.float32)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # We perform rounding because there might ba a minimal difference
            # even between two images of the same subject
            # (image compared against itself)
            # Allclose is used instead of array_equal to
            # support array of floats (if we remove rounding).
            movement = \
                not numpy.allclose(
                    numpy.rint(good_new),
                    numpy.rint(good_old)
                )
        except Exception as ex:
            LOGGER.error(
                f"Error during the execution of\
                     the optical flow estimation! [{ex}]")

        return movement
