import numpy as np
import tfBlitz
import os
from random import shuffle
import cv2
import random
import time

def addframes(data):
    sequence=2
    batch=[]
    length=len(data)
    length=length-(length%sequence)
    for i in range(length):
        if i%sequence==0:
            elelist=[]
            array = []
            label = []
            for elementlist in data[i:i + sequence]:
                array.append(elementlist[0])
                label.append(elementlist[1])
            label=np.array(label)
            array = np.array(array)
            elelist.append(array)
            for labels in label:
                elelist.append(labels)
            batch.append(np.array(elelist))

    print(len(batch))
    return batch

def processdata(datafilepath,savefilepath):

    firstloop=True

    paths=tfBlitz.getpaths(datafilepath,savefilepath,'npy','tfrecord')
    lenofdata=0

    print(len(paths))

    for path in paths:
        data = np.load(path, allow_pickle=True)  # slow
        lenofdata = lenofdata + len(data)

        wlabel = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        alabel = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        slabel = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        dlabel = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        walabel = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        wdlabel = [0, 0, 0, 0, 0, 1, 0, 0, 0]
        salabel = [0, 0, 0, 0, 0, 0, 1, 0, 0]
        sdlabel = [0, 0, 0, 0, 0, 0, 0, 1, 0]
        nulllabel = [0, 0, 0, 0, 0, 0, 0, 0, 1]

        w=[]
        a=[]
        s=[]
        d=[]
        sa=[]
        sd=[]
        wa=[]
        wd=[]
        null=[]

        data=labelclassiffier(data)
        # data=addframes(data)


        for elementlist in data:
            i = elementlist[-1]
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
            elif np.array_equal(i,salabel):
                sa.append(elementlist)
            elif np.array_equal(i,sdlabel):
                sd.append(elementlist)
            elif np.array_equal(i,nulllabel):
                null.append(elementlist)
            else:
                print('ERROR')



        shuffle(w)
        shuffle(s)
        shuffle(a)
        shuffle(d)
        shuffle(wa)
        shuffle(wd)
        shuffle(sa)
        shuffle(sd)
        shuffle(null)

        minlen1=int((len(sa)+len(sd)+len(s))/3)
        minlen2=int((len(sa)+len(sd))/2)
        w=w[:minlen1]
        wa=wa[:minlen1]
        wd=wd[:minlen1]
        sa=sa[:minlen2]
        sd=sd[:minlen2]
        a=a[:minlen2]
        d=d[:minlen2]
        s=s[:minlen2]
        null=null[:minlen2]
        data2=a+d+s+w+wd+wa+null+sa+sd

        shuffle(data2)

        if data2!=[]:
            elementlist=random.choice(data2)
            for i in elementlist:
                print(i.shape)

        else:
            print(None)

        # for i in data2:
        #     time.sleep(0.5)
        #     print(i[1:])
        #     for j in i[0]:
        #         cv2.imshow('win',j)
        #         cv2.waitKey(1000)

        print(len(data2))
        if len(data2)>10:
            name=os.path.basename(path)
            name=name[:-4]
            tfBlitz.tfwrite(f'{savefilepath}\\{name}',data2,firstloop)
            firstloop=False
            # np.save(f'{savefilepath}\\{name}',data2)
        print(lenofdata)
        print('------------------------------')


    return

def labelclassiffier(data):

    w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    a = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    s = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    null = [0, 0, 0, 0, 0, 0, 0, 0, 1]


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
        elif np.array_equal(i, [0, 1, 1, 0]):
            label=sa
        elif np.array_equal(i, [0, 0, 1, 1]):
            label=sd
        elif np.array_equal(i, [0, 0, 0, 0]):
            label=null
        else:
            print('OTHER')
            label=null

        label=np.array(label)
        data2.append([elementlist[0],label])

    return data2
