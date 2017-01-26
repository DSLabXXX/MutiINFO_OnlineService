import time
import zmq
import pprint
import json

def result_collector():
    context = zmq.Context()
    results_receiver = context.socket(zmq.PULL)
    results_receiver.bind("tcp://127.0.0.1:5558")
    collecter_data = {}
    collecter_data['num']=0
    collecter_data['error']=0
    count = 0
    while True:
        result = results_receiver.recv_json()
        encode_result = json.loads(result)
        if encode_result:
            print(encode_result)
            collecter_data['num'] += 1
        else:
            collecter_data['error'] += 1
        # if x == 999:
        #     pprint.pprint(collecter_data)
        count+=1
        pprint.pprint(collecter_data)
        print(count)

if __name__ == '__main__':
    result_collector()
