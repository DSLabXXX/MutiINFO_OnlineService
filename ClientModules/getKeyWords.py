# -*- coding:utf-8 -*-
from subprocess import Popen, PIPE, STDOUT
import os


def get_keywords(jarfile, jsonpath):
    # jarfile = './XXXX.jar'
    p = Popen(['java', '-jar', jarfile, jsonpath], stdout=PIPE, stderr=STDOUT)
    n = 0
    for line in p.stdout:
        if n == 2:
            # print(line)
            return line
            break
        n += 1

def proc_split(pre_str):
    ns = pre_str.decode(encoding='UTF-8')
    ns = ns.replace('[', '').replace(']', '').replace(',', '').replace('\n', '')
    ns = ns.split(' ')
    return ns


def sys_call_jar(java_file):
    os.system('java_file>out_put_file')


if __name__ == '__main__':
    jar_file = r'KEstr.jar'
    # testfile = "/home/c11kpy/dataset/crawler_data/AppleDaily/中古好屋/20160101---------No title---------1315312.txt"
    testfile = "/home/c11kpy/dataset/news/20111117藍不分區立委-學者弱勢出線-蔡英文：國民黨第一次想到-給予肯定.txt"
    s = get_keywords(jar_file, testfile)
    list_result = proc_split(s)
    # print(list_result)
