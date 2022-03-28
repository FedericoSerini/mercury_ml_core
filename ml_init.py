import os
import pika
import sys
from message.send_ml_end_workload import send_ml_end_workload

from main import init_train

symbol = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "DOGEUSDT", "NEOUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT", "EOSUSDT",
          "TRXUSDT", "ETCUSDT", "XLMUSDT", "ZECUSDT", "ADAUSDT", "QTUMUSDT", "DASHUSDT", "XMRUSDT", "BTTUSDT"]


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()

    channel.queue_declare(queue='ml-command-init')

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        for sim in symbol:
            init_train(ticker=sim)
        send_ml_end_workload()

    channel.basic_consume(queue='ml-command-init', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)