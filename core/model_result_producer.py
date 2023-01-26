import json
import uuid
import logging
from datetime import datetime

import pika

from core.model_result import ModelResult
from infrastructure.rabbitmq.rabbitmq_config_provider import RabbitMQConfigProvider

(
    rabbitmq_host,
    rabbitmq_username,
    rabbitmq_password,
) = RabbitMQConfigProvider.get_broker_config()

queue_name = RabbitMQConfigProvider.get_queue_names_config().rabbitmq_results_queue_name


class ModelResultProducer:
    def __init__(
            self,
            model_type,
    ):
        super().__init__()
        self.model_type = model_type

    def produce_message(
            self,
            result,
    ):
        if not isinstance(result, ModelResult):
            raise ValueError('result should be an instance of ModelResult')

        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=rabbitmq_host,
                virtual_host='/',
                credentials=pika.credentials.PlainCredentials(rabbitmq_username, rabbitmq_password)
            )
        )
        channel = connection.channel()

        channel.queue_declare(queue=queue_name, durable=True)

        body_str = json.dumps(result.__dict__)
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=body_str.encode('utf-8'),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            )
        )

        # ToDo should we log entire body?
        logging.info(f'Searching result sent to: {queue_name} - {body_str}')
