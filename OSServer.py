# -*- coding:utf-8 -*-
__author__ = 'c11tch'
from GraphMng import GraphMng
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from DumpDataset import *
import threading
import numpy as np
import copy
import logging
import logging.config
import zmq
import json
import pprint
import sched
import time


class OSServer:

    def __init__(self):
        self.dictItem2IID = {}
        self.dictIID2Item = {}
        self.dictTag2TID = {}
        self.dictTID2Tag = {}
        self.valItemsNum = 0
        self.valTagsNum = 0

        self.jobQueue = [Queue(100000), Queue(100000)]
        self.current_q = 0
        self.JQ_list = ['jobQueue', 'jobQueue2']
        """ For manage copy graph using by thread
            = graph obj =    -graph ogl-      - graph 1 -      - graph 2 -
            =   state   =    -  None   -      -  ready  -      - updating -
            = usr queue =    -  None   -      -queue has val-  - queue.empty()=TRUE-
        """
        self.graphs = []
        self.num_of_graphs = 2
        self.graphState = []
        self.graphsOn = False
        self.mergeConsumer = True

        """ 製作GM物件以管理Graph """
        self.GraphMng = GraphMng()

        """ start a process exec pool """
        self.pool = ThreadPoolExecutor(11)

        self.log = logging.getLogger('[OSS]')
        self.set_log_conf()
        np.set_printoptions(threshold=10000)    # 強制顯示列印內容(無'...')

        self.s = sched.scheduler(time.time, time.sleep)

    """---------------------------------- Vertex Augment ---------------------------------- """
    """ 若無此ID，加入dict中，以供後續查詢。有就當沒事 """
    def add_new_item(self, item_id):
        if item_id not in self.dictItem2IID:
            """ 讓儲存的數值改為0開始 以免發生'ValueError: row index exceeds matrix dimensions' """
            self.dictItem2IID[item_id] = self.valItemsNum
            self.dictIID2Item[self.valItemsNum] = item_id
            self.valItemsNum += 1
            self.GraphMng.add_new_item(item_id)    # 將使用者加入至GraphMng物件中

    def add_new_tag(self, tag_id):
        if tag_id not in self.dictTag2TID:
            self.dictTag2TID[tag_id] = self.valTagsNum
            self.dictTID2Tag[self.valTagsNum] = tag_id
            self.valTagsNum += 1
            self.GraphMng.add_new_tag(tag_id)    # 將標籤加入至GraphMng物件中

    """ ---------------------------------- Edge Augment ---------------------------------- """
    """ 先確認點是否都已存入dict中，再做加邊動作 """
    def add_edge_i2t(self, item_id, tag_id, rating):
        self.add_new_item(item_id)
        self.add_new_tag(tag_id)

        iid = self.dictItem2IID.get(item_id)
        tid = self.dictTag2TID.get(tag_id)
        self.GraphMng.add_edge_i2t(iid, tid, rating)

    """
    ---------------------------------- To Generate Graphs ----------------------------------
    """
    def make_graph_from_queue(self):
        udp_q = self.current_q
        if udp_q == 0:
            self.current_q = 1
        elif udp_q == 1:
            self.current_q = 0
        while not self.jobQueue[udp_q].empty():
            d_temp = self.jobQueue[udp_q].get()
            title = d_temp['Title']
            self.add_new_item(title)

            # title need -> ID
            for key in d_temp['TitleKey']:
                self.add_new_tag(key)
                self.add_edge_i2t(title, key, 1.2)

            for key in d_temp['KeyWord']:
                self.add_new_tag(key)
                self.add_edge_i2t(title, key, 0.6)
        """ 完成矩陣 """
        self.GraphMng.make_all_mat()
        self.GraphMng.upd_trans_mat()
        self.dump_info()
        self.dump_graph()

        # m = self.GraphMng.ColumnNomalization(self.GraphMng.matrixTrans)    # Column Normalization
        # m.tolil()[0,0] = 1    # 改變數值時 lil 格式比 csr格式有效率
        # print m.todense()
    """ --------------------------- End make_Graph_from_Data() ------------------------- """

    """  --------------------------- To dump and load graph -------------------------  """
    def dump_graph(self):
        data = (self.GraphMng.list_IT_tid, self.GraphMng.list_IT_iid, self.GraphMng.list_IT_rat,
                self.GraphMng.valTagsNum, self.GraphMng.valItemsNum)
        save(data, 'save/graph.pkl')
        self.log.info('Dumping Graph success')

    def load_graph(self):
        data = load('save/graph.pkl')
        (self.GraphMng.list_IT_tid, self.GraphMng.list_IT_iid, self.GraphMng.list_IT_rat,
         self.GraphMng.valTagsNum, self.GraphMng.valItemsNum) = data
        self.log.info('Loading Graph success')
        self.GraphMng.make_all_mat()
        self.GraphMng.upd_trans_mat()

    def dump_info(self):
        data = (self.dictItem2IID, self.dictIID2Item, self.dictTag2TID,
                self.dictTID2Tag, self.valItemsNum, self.valTagsNum)
        save(data, 'save/info.pkl')
        """ 另外存一份可以看的字典 """
        save_dict_to_json(self.dictIID2Item, 'save/items.json')
        save_dict_to_json(self.dictTID2Tag, 'save/tags.json')
        self.log.info('Dumping Base Information success')

    def load_info(self):
        data = load('save/info.pkl')
        (self.dictItem2IID, self.dictIID2Item, self.dictTag2TID,
         self.dictTID2Tag, self.valItemsNum, self.valTagsNum) = data
        self.log.info('Loading Base Information success')
        self.log.info('num of items:{0}'.format(self.valItemsNum))
        self.log.info('num of tags:{0}'.format(self.valTagsNum))
    """  --------------------------- End dump and load graph -------------------------  """

    """ --------------------------- Start Process Service ------------------------- """
    def start_process(self):
        self.log.info('startProcess')

        """ init graph """
        self.load_info()
        self.load_graph()

        """ copy graph obj to graphs list """
        for i in range(self.num_of_graphs):
            self.graphs.append(copy.copy(self.GraphMng))
            self.graphState.append('ready')

        """ Start a Server Socket """

        """ Bind host * and port 8888 """
        host_clwer = '192.168.4.213'
        host_oss = '192.168.4.216'
        host_web = '192.168.4.216'
        port = 8888
        url_router = "tcp://%s:%s" % (host_web, port)
        url_worker = 'inproc://ping-workers'
        worker_num = 10
        port = 5557
        portc = 5558
        url_server = "tcp://%s:%s" % (host_clwer, port)
        url_client = "tcp://%s:%s" % (host_oss, portc)

        context = zmq.Context()

        router = context.socket(zmq.ROUTER)
        router.bind(url_router)

        workers = context.socket(zmq.DEALER)
        workers.bind(url_worker)

        """ Define worker's job """
        def worker(name, url_worker, context):
            self.log.warning('worker {0} start'.format(name))
            """ connect to router """
            worker = context.socket(zmq.REP)
            worker.connect(url_worker)
            while True:
                try:
                    """ Receive a JSON encode dict contains { job ID , REQ , DATA } """
                    request_d = worker.recv_json()
                    request_dict = json.loads(request_d)
                    id, req_type, data = (request_dict['ID'], request_dict['REQ'], request_dict['DATA'])
                    self.log.info('-' * 80)
                    if len(data) <= 4:
                        self.log.info('[worker]worker {0} recv job ID:{1} REQ:{2} DATA:{3}'.format(name, id, req_type, data))
                    else:
                        self.log.info('[worker]worker {0} recv job ID:{1} REQ:{2} '
                                      'DATA: Data is too long to show'.format(name, id, req_type))
                    """ init reply task dict """
                    reply_task = {
                        'ID': id,
                        'REQ': req_type,
                    }
                    self.log.info('-' * 80)

                    if req_type == 'R':
                        """ job REQ R is Recommendation """
                        if len(data) == 3 and isinstance(data[0], list) and isinstance(data[1], list) \
                                and isinstance(data[2], int):
                            result = self.recom_task(data)
                            # 觀看query向量
                            self.log.warning('query items: {0}'.format(self.analyze_result(data[0], 1)))
                            self.log.warning('query tags: {0}'.format(self.analyze_result(data[1], 2)))
                            reply = self.analyze_result(result, data[2])
                        else:
                            reply = 'Data format error. [[item], [tag], rlt type] ' \
                                    'your data is: {0}'.format(data)
                    elif req_type == 'U':
                        """ job REQ U is Update graphs """
                        self.upd_mat_request()
                        reply = '鳩咪'
                    else:
                        reply = 'What are u doing!!!!!!'
                    """ reply task message to client """
                    reply_task['RESULT'] = reply
                    task_json_encode = json.dumps(reply_task)
                    worker.send_json(task_json_encode)
                    self.log.info('[startProcess]worker {0} reply job{1} : {2}'.format(name, id, reply_task))
                    self.log.info('-' * 80)
                except TypeError as err:
                    self.log.error('worker {0} TypeError : {1}'.format((name, err)))
                    break
                except:
                    self.log.error('worker {0} error'.format(name))
                    break
            worker.close()

        """ Define consumer's job, pull data and push to OSServer"""
        def consumer(url_s, url_c):
            import random
            consumer_id = random.randrange(1, 10005)
            self.log.warning("Start consumer #{0}".format(consumer_id))
            context = zmq.Context()
            # receive work
            consumer_receiver = context.socket(zmq.PULL)
            consumer_receiver.connect(url_s)
            # send work
            consumer_sender = context.socket(zmq.PUSH)
            consumer_sender.connect(url_c)
            num = 0
            while True:
                num += 1
                job = consumer_receiver.recv_json()
                consumer_sender.send_json(job)
                if num % 100 == 0:
                    self.log.info('from worker #{0} push {1} tasks to service.'.format(consumer_id, num))

        """ Define result collector's job """
        def result_collector(url_c):
            self.log.warning("Start result collector")
            context = zmq.Context()
            results_receiver = context.socket(zmq.PULL)
            results_receiver.bind(url_c)
            # results_receiver.bind(url_router)
            collector_data = dict()
            collector_data['num'] = 0
            collector_data['error'] = 0

            while True:
                try:
                    """ Receive a JSON encode dict contains { 'TitleKey', 'KeyWord', ... } """
                    encode_result = results_receiver.recv_json()
                    # encode_result = json.loads(result)
                    if encode_result:
                        self.log.info('TitleKey:{0} ; KeyWord:{1}'.format(encode_result['TitleKey'],
                                                                          encode_result['KeyWord']))
                        self.jobQueue[self.current_q].put(encode_result)
                        self.log.info('put job in queue#{0}'.format(self.current_q))
                        collector_data['num'] += 1
                        # if collector_data['num'] >= 10:
                        #     break
                    else:
                        collector_data['error'] += 1
                    if collector_data['num'] % 100 == 0:
                        pprint.pprint(collector_data)
                except IOError as err:
                    self.log.debug('ERROR {0}'.format(err))
            results_receiver.close()
            context.term()

        """ start workers """
        for i in range(worker_num):
            thread = threading.Thread(target=worker, args=(i, url_worker, context))
            thread.start()
        """ start consumers """
        if self.mergeConsumer:
            for i in range(3):
                thread = threading.Thread(target=consumer, args=(url_server, url_client))
                thread.start()
        """ start result collector """
        thread = threading.Thread(target=result_collector, args=(url_client,))
        thread.start()
        # 每個小時(3600)更新Graph一次
        self.auto_upd_graph(3600)
        # End auto upd
        zmq.device(zmq.QUEUE, router, workers)
        router.close()
        workers.close()
        context.term()
    """ ------------------------------- End startProcess() ----------------------------- """

    """ ------------------------------- Update Graph Object ----------------------------- """
    def perform(self, inc):
        """ Using sched to instead infinite loop """
        self.s.enter(inc, 0, self.perform, (inc,))
        self.upd_mat_request()
        self.s.run()

    def auto_upd_graph(self, inc=3600):
        """ Calling perform using thread """
        thread = threading.Thread(target=self.perform, args=(inc,))
        thread.start()

    def upd_mat_request(self):
        """ update all graph from original to g2 to g1 """
        self.make_graph_from_queue()

        """ Update latest graph first g2 > g1 """
        opposite_graphs = [i for i in range(self.num_of_graphs)]
        opposite_graphs.reverse()
        for gIdx in opposite_graphs:
            if self.graphState[gIdx] == 'ready':
                """ Change the flag to updating """
                self.graphState[gIdx] = 'updating'

                while True:
                    """ Waiting for all threads who using this graph exec finished """
                    if self.graphs[gIdx].userQueue.empty():
                        """ updating current graph """
                        self.graphs[gIdx] = copy.copy(self.GraphMng)
                        self.log.warning('update graph {0}'.format(gIdx))
                        break
                """ Update finished and graph reopen """
                self.graphState[gIdx] = 'ready'
    """ ------------------------------- End Update Graph Object ----------------------------- """

    def analyze_result(self, rlt_index, rlt_type):
        self.log.debug('[analyze_result] resultType = {0}'.format(rlt_type))
        if rlt_type == 1:
            self.log.debug('[analyze_result] recommend items : ')
            result = [self.dictIID2Item[index] for index in rlt_index]
        else:
            self.log.debug('[analyze_result] recommend tags : ')
            result = [self.dictTID2Tag[index] for index in rlt_index]
        return result

    def recom_task(self, lst_job):
        lst_iid, lst_tid, rlt_type = (lst_job[0], lst_job[1], lst_job[2])
        """ 比對代表數值 """
        vec_query_iid = np.array(lst_iid)
        # vec_query_iid = [self.dictItem2IID[index] for index in vec_query_iid]
        vec_query_tid = np.array(lst_tid)
        # vec_query_tid = [self.dictTag2TID[index] for index in vec_query_tid]
        norm_type = 1
        k = 10
        t = True
        while t:
            for gIdx in range(len(self.graphState)):
                """ Checking graph is ready """
                if self.graphState[gIdx] == 'ready':
                    """ using userQueue to tell graph updater some thread using this graph """
                    self.graphs[gIdx].userQueue.put(1)
                    """ Process PageRank """
                    rlt_idx = self.graphs[gIdx].exec_ranking_process(vec_query_iid, vec_query_tid,
                                                                     rlt_type, norm_type, k)
                    """ For Check threads are synchronize """
                    self.log.info('[recom_task] use graph {0} now time= {1}'.format(gIdx, time.localtime(time.time())))
                    t = False
                    self.graphs[gIdx].userQueue.get()
                    break
        """ 為了測試 thread 拖時間 """
        # MATRIX_SIZE = 1000
        # a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)
        # b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE)
        # c_cpu = np.dot(a_cpu, b_cpu)

        return rlt_idx
    """ End recom for thread version """

    def set_log_conf(self):
        self.log.setLevel(logging.DEBUG)

        file_hdlr = logging.FileHandler('log/OSServer.log')
        file_hdlr.setLevel(logging.DEBUG)

        console_hdlr = logging.StreamHandler()
        console_hdlr.setLevel(logging.INFO)

        formatter = logging.Formatter('%(levelname)-8s - %(asctime)s - %(name)-12s - %(message)s')
        file_hdlr.setFormatter(formatter)
        console_hdlr.setFormatter(formatter)

        self.log.addHandler(file_hdlr)
        self.log.addHandler(console_hdlr)


if __name__ == '__main__':
    app = OSServer()
    app.start_process()
