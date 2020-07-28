import numpy as np
import glob
import os
from random import shuffle
import cv2
import random
import pandas as pd
import time
import mytime

def addframes(data):
    batch=[]
    for i in range(len(data) - 1):
        if i % 2 == 0:
            array = []
            label = []
            for elementlist in data[i:i + 2]:
                array.append(elementlist[0])
                label.append(elementlist[1])
            label=np.array(label)
            # print(label.shape)
            array = np.array(array)
            # print(array.shape)
            batch.append(np.array([array,label]))

    print(len(batch))
    return batch

def processdata(datafilepath,savefilepath):

    paths=mytime.getpaths(datafilepath,savefilepath,'npy')
    lenofdata=0

    print(len(paths))

    for path in paths:
        data = np.load(path, allow_pickle=True)  # slow
        lenofdata = lenofdata + len(data)

        wlabel =  [1,0,0,0,0,0,0]
        alabel =  [0,1,0,0,0,0,0]
        slabel =  [0,0,1,0,0,0,0]
        dlabel =  [0,0,0,1,0,0,0]
        walabel = [0,0,0,0,1,0,0]
        wdlabel = [0,0,0,0,0,1,0]
        nulllabel =[0,0,0,0,0,0,1]

        w=[]
        a=[]
        s=[]
        d=[]
        wa=[]
        wd=[]
        null=[]

        data=labelclassiffier(data)
        data=addframes(data)
        labelindex=2

        for elementlist in data:
            i = elementlist[1][1]
            if np.array_equal(i,wlabel):
                w.append(elementlist)
            elif np.array_equal(i,alabel):
                a.append(elementlist)
            elif np.array_equal(i,slabel):
                s.append(elementlist)
            elif np.array_equal(i,dlabel):
                d.append(elementlist)
            elif np.array_equal(i,walabel):
                wa.append(elementlist)
            elif np.array_equal(i,wdlabel):
                wd.append(elementlist)
            # elif np.array_equal(i,sa):
            #     sa.append(elementlist)
            # elif np.array_equal(i,sd):
            #     sd.append(elementlist)
            elif np.array_equal(i,nulllabel):
                null.append(elementlist)


        shuffle(w)
        shuffle(s)
        shuffle(a)
        shuffle(d)
        shuffle(wa)
        shuffle(wd)
        shuffle(null)

        minlen=len(s)
        w=w[:minlen]
        wa=wa[:minlen]
        wd=wd[:minlen]
        a=a[:minlen]
        d=d[:minlen]
        null=null[:minlen]
        data2=a+d+s+w+wd+wa+null

        shuffle(data2)

        if data2!=[]:
            elementlist=random.choice(data2)
            for i in elementlist:
                print(i.shape)
            # df=pd.DataFrame(elementlist)
            # print(df)

        else:
            print(None)

        print(len(data2))
        if len(data2)>10:
            name=os.path.basename(path)
            np.save(f'{savefilepath}\\{name}',data2)
        print(lenofdata)
        print('------------------------------')


    return

def labelclassiffier(data):

    w = [1, 0, 0, 0, 0, 0, 0]
    a = [0, 1, 0, 0, 0, 0, 0]
    s = [0, 0, 1, 0, 0, 0, 0]
    d = [0, 0, 0, 1, 0, 0, 0]
    wa = [0, 0, 0, 0, 1, 0, 0]
    wd = [0, 0, 0, 0, 0, 1, 0]
    null = [0, 0, 0, 0, 0, 0, 1]
    none=[0,0,0,0,0,0,0]

    data2=[]
    for elementlist in data:

        i = elementlist[1]

        if np.array_equal(i, [1, 0, 0, 0]):
            label=w
        elif np.array_equal(i, [0, 1, 0, 0]):
            label=a
        elif np.array_equal(i, [0, 0, 1, 0]):
            label=s
        elif np.array_equal(i, [0, 0, 0, 1]):
            label=d
        elif np.array_equal(i, [1, 1, 0, 0]):
            label=wa
        elif np.array_equal(i, [1, 0, 0, 1]):
            label=wd
        # elif np.array_equal(i, [0, 1, 1, 0]):
        #     label=sa
        # elif np.array_equal(i, [0, 0, 1, 1]):
        #     label=sd
        elif np.array_equal(i, [0, 0, 0, 0]):
            label=null
        else:
            label=none
        label=np.array(label)
        # print(label)
        data2.append([elementlist[0],label])

    # print(len(data2))
    return data2
