import pickle
from kafka import KafkaConsumer
from common.Logger import LOGGER


class Consumer:
    def __init__(
            self,
            endpoints=None,
            group_id="default",
            consumer_timeout_ms=250
    ) -> None:

        if endpoints is None:
            endpoints = ["localhost:9092"]
        self.group_id = group_id
        self.consumer_timeout_ms = consumer_timeout_ms
        self.disconnected = False

        LOGGER.debug("Creating Kafka consumer")
        self.consumer = KafkaConsumer(
            value_deserializer=lambda v: pickle.loads(v),
            group_id=self.group_id,
            bootstrap_servers=endpoints,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            max_poll_interval_ms=10000,
            consumer_timeout_ms=self.consumer_timeout_ms
        )

    def get_topics(self):
        return self.consumer.topics()

    def subscribe(self, topic):
        for t in topic:
            try:
                LOGGER.debug(f"Subscribing to topics {t}.")
                self.consumer.subscribe(t)
            except Exception as e:
                LOGGER.error(
                    f"Unable to subscribe to topic {topic}. [{e}]")

    def consume_messages(self):
        try:
            LOGGER.debug("Reading messages")
            return list(map(lambda message: message.value, self.consumer))

        except Exception as e:
            LOGGER.error(
                f"Error while reading messages from Kafka broker:  [{e}]")
            return []

    def consume_messages_poll(self):
        data = []
        try:
            LOGGER.debug("Reading messages using poll")
            messages = self.consumer.poll(timeout_ms=1000)
            for key, record in messages.items():
                for item in record:
                    data.append(item.value)

            return data

        except Exception as e:
            LOGGER.error(
                f"Error while polling messages from Kafka broker: [{e}]")
            return []

    def pause(self):
        pass

    def resume(self):
        pass

    def close(self):
        LOGGER.debug("Shutting down consumer")
        self.disconnected = True
        self.consumer.close()
