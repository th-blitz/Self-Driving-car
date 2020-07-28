import datetime
import time
import glob
import os

def getdateandtime(timestamp_in_sec=None):
    if timestamp_in_sec==None:
        timestamp_in_sec=time.time()
    dateandtime = str(datetime.datetime.fromtimestamp(timestamp_in_sec))
    dateandtime = list(dateandtime)
    dateandtime[10] = '_'
    dateandtime[13] = '-'
    dateandtime[16] = '-'
    dateandtime = dateandtime[:19]
    string = ''
    for i in dateandtime:
        string = string + i
    return string

def getpaths(from_directory,to_directory,filetype):
    frompaths=glob.glob(f'{from_directory}\\*.{filetype}')
    topaths=glob.glob(f'{to_directory}\\*.{filetype}')

    for path in topaths:
        try:
            frompaths.remove(path)
        except:
            pass

    return frompaths

