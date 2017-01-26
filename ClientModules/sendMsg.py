import time
import zmq
import json

def sendjson(json_file):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.bind("tcp://192.168.4.216:5557")
    zmq_socket.send_json(json_file)


if __name__ == '__main__':
    pass
    # while True:
    #     try:
    #         time.sleep(3)
    #         sendjson()
    #     except:
    #         print('error')
