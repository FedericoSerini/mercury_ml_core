import pika


def send_ml_end_workload():
    connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()

    channel.queue_declare(queue='end-ml-workload')

    channel.basic_publish(exchange='', routing_key='end-ml-workload', body='')
    print(" [x] Sent 'end-ml-workload'")
    connection.close()
