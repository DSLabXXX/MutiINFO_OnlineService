# -*- coding:utf-8 -*-
import os
import json
from sendMsg import sendjson
import time
from getKeyWords import *


def save(data, file_name, path):
    destination = path + "/" + file_name
    f = open(destination, 'w')
    f.write(data)
    f.close()
    print("The code is saved.")


def load(file_name, path):
    if file_name != '':
        source = path + "/" + file_name
    else:
        source = path

    if os.path.exists(source):
        f = open(source, 'rb')
        print(source)
        data = []
        for i in f:
            # data.append(i.replace('\n', '').split(split_unit))
            try:
                data.append(i.decode('utf-8').replace('\n', '').replace('\r', ''))
            except:
                try:
                    data.append(i.decode('big5').replace('\n', '').replace('\r', ''))
                except:
                    try:
                        data.append(i.decode('x-windows-950').replace('\n', '').replace('\r', ''))
                    except:
                        data.append(i.decode('ISO-8859-1').replace('\n', '').replace('\r', ''))

            # 測試能否讀出
            # print(json.loads(data[0]))

            # ----------------------- 建立關鍵字 -----------------------
            # s = get_keywords(r'KE.jar', path) # 讀檔用法
            # jsonFile = json.loads(data[0])
            #
            # # 建立關鍵字 for 'Text' of News
            # str_context = jsonFile['Text']
            # s = get_keywords(r'KEstr.jar', str_context)
            # list_text_result = proc_split(s)
            #
            # # 建立關鍵字 for 'Title' of News
            # str_title = jsonFile['Title']
            # s = get_keywords(r'KEstr.jar', str_title)
            # list_title_result = proc_split(s)
            #
            # # 新增關鍵字to dict 並存回 json 檔案中
            # jsonFile['KeyWord'] = list_text_result
            # jsonFile['TitleKey'] = list_title_result
            # fw = open(source, 'w')
            # json.dump(jsonFile, fw, ensure_ascii=False)
            # fw.close()
            # print('key word isssssssssssssss{0} ; {1}'.format(list_title_result, list_text_result))
            # ----------------------- End keyword extract -----------------------

        f.close()
        # return data
        return data[0]
    else:
        # print("Load denied!!")
        None


def load_all(doc_names, data, path):
    if os.path.exists(path):
        if os.path.isdir(path):
            # print("path : " + path)
            filelist = os.listdir(path)

            for f in filelist:
                load_all(doc_names, data, os.path.join(path, f))

        elif os.path.isfile(path):
            # input()
            time.sleep(1)
            buffer = load('', path)
            if len(buffer):

                # simulate sent json file
                sendjson(buffer)

                data.append(buffer)
                doc_names.append(path)
                # print("file : " + path)
                # print(data[-1])
        else:
            print("Loading error!!")
    else:
        print("Load denied!!")

if __name__ == '__main__':
    filelist = []
    data = []
    load_all(filelist, data, r'/home/c11kpy/dataset/crawler_data/')
