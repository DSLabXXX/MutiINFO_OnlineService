__author__ = 'c11tch'
import pickle
import logging
import logging.config
import json

log = logging.getLogger('[OSS].[DD]')


def save(data, filename='save/default.pkl'):
    fh = None
    try:
        fh = open(filename, 'wb')
        pickle.dump(data, fh)
        fh.close()
    except(EnvironmentError, pickle.PickleError) as err:
        log.error(err)
    finally:
        if fh is not None:
            fh.close()


def load(filename='save/default.pkl'):
    fh = None
    try:
        fh = open(filename, 'rb')
        data = pickle.load(fh)
        fh.close()
        return data
    except(EnvironmentError, pickle.PickleError) as err:
        log.error(err)
    finally:
        if fh is not None:
            fh.close()


def save_dict_to_json(obj_dict, filename='save/default.json'):
    f = open(filename, 'w')
    json.dump(obj_dict, f, ensure_ascii=False)
