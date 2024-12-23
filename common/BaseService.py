from abc import ABC, abstractmethod
from os.path import isfile, join

import os
import pickle
import queue
import threading
import time
import traceback
import uuid

from common.Logger import LOGGER
from common.Get_Param import get_parameters
from common.kafka.kafka_manager import KafkaConsumerThread, \
    KafkaProducer
from distutils import util
from collections import deque

POLL_INTERVAL = 0.01


class BaseService(ABC):

    def __init__(
            self,
            consumer_topics=None,
            producer_topics=None,
            models=None,
            timeout=POLL_INTERVAL,
            asynchronous=False
    ):

        if models is None:
            models = []
        if producer_topics is None:
            producer_topics = []
        if consumer_topics is None:
            consumer_topics = []
        self.asynchronous = asynchronous
        self.models = models

        if len(self.models) > 1:
            LOGGER.info("Provided multiple AI executors")
            assert all(isinstance(x, type(models[0])) for x in self.models), \
                "AI executors mixing is not supported!"

        self.batch_size = int(get_parameters('batch_size', 1))

        if self.asynchronous:
            LOGGER.info("Create queue for asynchronous processing.")
            self.sync_queue = deque()
        else:
            LOGGER.info("Create queue for synchronous processing.")
            self.accumulator = queue.Queue(maxsize=self.batch_size)

        # These variables are used to control the data ingestion rate
        # when processing a video.
        self.ingestion_rate = 0
        self.processing_rate = 1
        self.last_put = 0
        self.last_fetch = 0

        self.timeout = timeout

        self._init_kafka_connection(consumer_topics, producer_topics)

        self.stop_thread = False
        self._post_init()

    def _init_kafka_connection(self, consumer_topics, producer_topics):
        """
        Initializes the Kafka backend, if enabled.
        """

        self.kafka_enabled = bool(
            util.strtobool(get_parameters("KAFKA_ENABLED", "True"))
        )

        if self.kafka_enabled:
            LOGGER.debug("Kafka support enabled in YAML file")
            self.kafka_address = get_parameters("KAFKA_BIND_ENDPOINTS", "127.0.0.1:9092")
            if isinstance(self.kafka_address, str):
                LOGGER.debug(
                    f"Parsing string received from K8s controller \
                        {self.kafka_address}"
                )
                self.kafka_address = self.kafka_address.split("|")
            if producer_topics:
                self.producer = KafkaProducer(
                    self.kafka_address,
                    topic=producer_topics,
                    asynchronous=self.asynchronous
                )
            if consumer_topics:
                self.consumer = KafkaConsumerThread(
                    self.kafka_address,
                    topic=consumer_topics,
                    callback=self.put
                )

            LOGGER.info(
                f"Connection to Kafka broker/s \
                    {self.kafka_address}completed."
            )
            LOGGER.info(f"Consumer topics are {consumer_topics}.")
            LOGGER.info(f"Producer topics are {producer_topics}.")

        return

    def _post_init(self):
        """
        It starts the main data acquisition loop in a separate thread. It can
        be overridden to add some post initialization calls.
        """
        threading.Thread(target=self.fetch_data, daemon=True).start()
        return

    def put(self, data):
        """
        Call this function to push data into the component. For example,
        after you extract a frame from video stream, you can call put(image).
        Depending on the value of the 'asynchronous' parameter, the data will
        be put into a different data structure.
        """
        data = self.preprocess(data)
        if data:
            if self.asynchronous:
                return self._put_data_asynchronous(data)
            else:
                return self._put_data_synchronous(data)

    def fetch_data(self):
        if self.asynchronous:
            self._fetch_asynchronous()
        else:
            self._fetch_synchronous()

    def _fetch_synchronous(self):
        LOGGER.info("Start synchronous fetch loop.")
        while not self.stop_thread:
            if self.accumulator.full():
                self._extract_wrapper_sync(self.batch_size)
            elif self.accumulator.qsize() > 0 and \
                    (time.time() - self.last_fetch > self.timeout):
                LOGGER.info(
                    "Timeout reached. \
                        Processing and flushing the \
                            remaining elements of the queue."
                )
                self._extract_wrapper_sync(self.accumulator.qsize())
            else:
                time.sleep(POLL_INTERVAL)

    def _fetch_asynchronous(self):
        LOGGER.info("Start asynchronous fetch loop.")
        while not self.stop_thread:
            total_stored_elements = len(self.sync_queue)
            if total_stored_elements >= self.batch_size:
                try:
                    self._extract_wrapper_async(self.batch_size)
                except Exception as e:
                    LOGGER.error(f"Error processing received data: {e}")
                    traceback.print_exc()
            # if we don't receive data for n seconds, flush the queue
            elif total_stored_elements > 0 and \
                    (time.time() - self.last_fetch > self.timeout):
                LOGGER.info(
                    "Timeout reached. \
                        Processing and flushing the \
                            remaining elements of the queue."
                )
                self._extract_wrapper_async(total_stored_elements)
            else:
                time.sleep(POLL_INTERVAL)

    def _extract_wrapper_async(self, amount):
        token = [self.sync_queue.popleft() for _ in range(amount)]
        self.last_fetch = time.time()
        self.distribute_data(self.flatten(token))
        self.processing_rate = amount / (time.time() - self.last_fetch)
        LOGGER.debug(
            f"Data Processing Speed: {self.processing_rate} objects/s"
        )

    def _extract_wrapper_sync(self, amount):
        token = [self.accumulator.get() for _ in range(amount)]
        self.last_fetch = time.time()
        self.distribute_data(self.flatten(token))
        [self.accumulator.task_done() for _ in range(amount)]
        self.accumulator.join()

    def _put_data_asynchronous(self, data):
        self.sync_queue.append(data)
        self.last_put = time.time()
        return

    def _put_data_synchronous(self, data):
        self.accumulator.put(data)

        if self.accumulator.full():
            token = [self.accumulator.get() for _ in range(self.batch_size)]
            self.distribute_data(self.flatten(token))
            [self.accumulator.task_done() for _ in range(self.batch_size)]

            self.accumulator.join()

        return

    def flatten(self, S):
        if not S:
            return S
        if isinstance(S[0], list):
            return self.flatten(S[0]) + self.flatten(S[1:])
        return S[:1] + self.flatten(S[1:])

    # Distributes data in the queue to models associated to this service
    def distribute_data(self, data=[], **kwargs):
        """
        This function sends the data to all the AI models passed to with this
        component during the initialization phase.
        """
        for ai in self.models:
            self.process_data(ai, data)
        return

    @abstractmethod
    def process_data(self, ai, data, **kwargs):
        """
        The user needs to implement this function to call the main processing
        function of the AI model and decide what to do with the result.
        """
        return

    def preprocess(self, data, **kwargs):
        """
        The user can override this function to inject some preprocessing
        operation to be executed before the data is added to the data
        structure by the 'put()' function.
        """
        return data


class FileOperations:
    """
    Class containing file operations to read/write from disk.
    """

    def read_from_disk(self, path):
        data = []
        try:
            with open(path, 'rb') as diskdata:
                data = pickle.load(diskdata)
        except Exception as ex:
            LOGGER.error(f"Unable to read or load the file! {ex}")

        return data

    def read_from_redis(self, key, r):
        data = []
        try:
            data_bytes = r.get(key)
            data = pickle.loads(data_bytes)
        except Exception as ex:
            LOGGER.error(f"Unable to read or load the file! {ex}")

        return data

    def delete_from_disk(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        else:
            LOGGER.error("The file does not exist.")

    def write_to_disk(self, data, folder, exts=".dat"):
        isExist = os.path.exists(folder)
        if not isExist:
            os.makedirs(folder)

        filename = str(uuid.uuid1())
        with open(f"{folder}{filename}{exts}", 'ab') as result:
            pickle.dump(data, result)

    def get_files_list(self, folder):
        file_list = []
        for f in os.listdir(folder):
            if isfile(join(folder, f)):
                file_list.append(join(folder, f))
        return file_list
