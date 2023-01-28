import sys
import traceback
import datetime
import re


class Logger:
    def __init__(self, filename='log_errors'):
        """"""
        self.filename = f'{filename}.txt'

    def log_date(self):
        log = open(self.filename, 'a+')
        print("\n-----------------------\nException date time: {}".format(datetime.datetime.now()), file=log)
        log.close()

    def log_patient_id(self, path):
        log = open(self.filename, 'a+')
        self.log_date()
        print('Processing path: %s\n' % path, file=log)
        log.close()

    def log(self, e, event, dsc_mrp=None, dce_mra=None):
        path = ''
        if 'dsc' in event:
            path = dsc_mrp.npy_folder if dsc_mrp.npy_folder is not None else ''
        elif 'dce' in event:
            path = dce_mra.npy_folder if dce_mra.npy_folder is not None else ''
        if bool(re.match('^[a-zA-Z0-9- :/]*$', path)):
            self.log_patient_id(path)
        else:
            print(path)

        log = open(self.filename, 'a+')
        event = 'Event: %s\n' % event.replace(u'\u2713', '+')
        if bool(re.match('^[a-zA-Z0-9- :/]*$', path)):
            print(event, file=log)
            traceback.print_exc(file=log)
        else:
            print(event)
        traceback.print_exc()
        print(e)
        log.close()
