from threading import Thread
from common.kafka.consumer import Consumer
from common.kafka.producer import Producer


class KafkaProducer:
    def __init__(self, endpoints, topic=None, asynchronous=False):
        if topic is None:
            topic = []
        self.producer = Producer(endpoints=endpoints)
        self.topic = topic
        self.asynchronous = asynchronous

    def write_result(self, data):
        if self.asynchronous:
            return self.producer.publish_data_asynchronous(
                data, topic=self.topic)
        else:
            return self.producer.publish_data_synchronous(
                data, topic=self.topic)


class KafkaConsumerThread(Thread):
    def __init__(self, endpoints, topic=None, callback=None):
        super().__init__()
        if topic is None:
            topic = []
        self.consumer = Consumer(endpoints=endpoints)
        self.callback = callback
        self.topic = topic

        # We do this before actually reading from the topic
        self.consumer.subscribe(self.topic)

        self.daemon = True
        self.start()

    def run(self):
        while not self.consumer.disconnected:
            data = self.consumer.consume_messages_poll()
            if data:
                self.callback(data)
