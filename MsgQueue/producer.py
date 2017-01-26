import time
import zmq
import json

def producer():
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.bind("tcp://127.0.0.1:5557")
    # Start your result manager and workers before you start your producers
    for num in range(50):
        work_message = {'num': num}
        task = {
            'Topic':num,
            'Category': 'social',
            'Source': 'apple',
            'Time': '{0}'.format(time.localtime(time.time())),
            'Location': 'Your home',
            'People': 'Ma',
            'Keyword': [1,2,3,4,5],
            'stand': 90,
            'Url': 'www.mememe.tw',
            'Imgurl': 'www.imgurl.tw',
            'Hdfsurl': '/home/adsad/Ya'
        }
        task_json = json.dumps(task)
        # zmq_socket.send_json(work_message)
        zmq_socket.send_json(task_json)
        if num%1000 == 0:
            print('{0}'.format(num))

if __name__ == '__main__':
    while True:
        try:
            time.sleep(3)
            producer()
        except:
            print('error')
