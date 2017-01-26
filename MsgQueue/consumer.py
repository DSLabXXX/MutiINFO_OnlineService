import time
import zmq
import random
import threading


def consumer():
    consumer_id = random.randrange(1, 10005)
    print("I am consumer #{0}".format(consumer_id))
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://127.0.0.1:5557")
    # send work
    consumer_sender = context.socket(zmq.PUSH)
    consumer_sender.connect("tcp://127.0.0.1:5558")
    num = 0
    while True:
        num += 1
        job = consumer_receiver.recv_json()
        consumer_sender.send_json(job)
        if num % 100 == 0:
            print('from {0} push {1} tasks to service.'.format(consumer_id, num))


if __name__ == '__main__':
    for i in range(3):
        thread = threading.Thread(target=consumer)
        thread.start()

