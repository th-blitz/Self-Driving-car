import numpy as np
import os
import cv2
import keys
import time
import glob
import tfBlitz


def getlabelinfo(datafilepath):

    wlist=0
    alist=0
    dlist=0
    slist=0
    walist=0
    wdlist=0
    salist=0
    sdlist=0
    nulllist=0

    lenofdata=0
    paths=glob.glob(f'{datafilepath}\\*.npy')
    for datapath in paths:

        data = np.load(datapath, allow_pickle=True)#slow

        lenofdata=lenofdata+len(data)

        for elementlist in data:
            i=elementlist[2]
            if np.array_equal(i,[1,0,0,0,0,0,0]):
                wlist=wlist+1
            elif np.array_equal(i,[0,1,0,0,0,0,0]):
                alist=alist+1
            elif np.array_equal(i,[0,0,1,0,0,0,0]):
                slist = slist + 1
            elif np.array_equal(i,[0,0,0,1,0,0,0]):
                dlist = dlist + 1
            elif np.array_equal(i,[0,0,0,0,1,0,0]):
                walist = walist + 1
            elif np.array_equal(i,[0,0,0,0,0,1,0]):
                wdlist = wdlist + 1
            elif np.array_equal(i,[0,0,0,0,0,0,0]):
                salist = salist + 1
            elif np.array_equal(i,[0,0,0,0,0,0,0]):
                sdlist = sdlist + 1
            elif np.array_equal(i,[0,0,0,0,0,0,1]):
                nulllist = nulllist + 1
            else:
                print('ERROR')

        print('------------------------------------')
        print(lenofdata)

        print(wlist)
        print(alist)
        print(slist)
        print(dlist)
        print(walist)
        print(wdlist)
        print(salist)
        print(sdlist)
        print(nulllist)

    listing = {
        'w': wlist,
        'a': alist,
        's': slist,
        'd': dlist,
        'wa': walist,
        'wd': wdlist,
        'sa': salist,
        'sd' : sdlist,
        'none':nulllist
    }
    print(listing)
    return

def start2(datafilepath,savefilepath,defSPEED=25,tweaks=None,printable=False):

    if tweaks==None:
        tweaks={
            'pausekey':'spacebar',
            'playbackkey':'a',
            'playforwardkey':'d',
            'fastkey':'w',
            'slowkey':'s',
            'markkey':'m',
            'prescisionkey':'p',
            'skipkey':'0',
            'skipviewedits':'9'
        }

    pausekey = tweaks['pausekey']
    playbackkey = tweaks['playbackkey']
    playforwardkey = tweaks['playforwardkey']
    fastkey = tweaks['fastkey']
    slowkey = tweaks['slowkey']
    markkey = tweaks['markkey']
    prescisionkey = tweaks['prescisionkey']
    skipkey=tweaks['skipkey']
    skipviewkey=tweaks['skipviewedits']

    paths=glob.glob(f'{datafilepath}\\*.npy')

    try:

        with open(f'{savefilepath}\\cleanedinfo.txt', 'r') as f:
            for line in f:
                stripped_line = line.strip()
                try:
                    paths.remove(stripped_line)
                except:
                    pass

    except:
        pass

    print('files left')
    print(len(paths))

    for path in paths:

        data = np.load(path, allow_pickle=True)

        while (True):


            print('=================================================================================================')
            print(len(data))
            time.sleep(2)
            SPEED = defSPEED
            elementindex = 0
            pause = False
            prescision = False
            markings = []
            lenofdata = len(data)
            mlist = []
            prev = -1
            loops = 0
            delayloop = 10
            redo = False
            prevp = 0
            prevp1 = 0
            skip = False

            while (True):

                loops = loops + 1
                playback = False
                playforward = False
                fast = False
                slow = False

                if keys.ifpressed(skipkey):
                    skip = True
                    break

                elementlist = data[elementindex]

                if printable == True:
                    if elementindex != prevp:
                        print(
                            '------------------------------------------------------------------------------------------------')
                        print(elementindex)
                        prevp = elementindex

                for i in elementlist:
                    if len(i.shape) <= 1 and printable == True:
                        if elementindex != prevp1:
                            prevp1 = elementindex
                            print(i)
                    if len(i.shape) > 1:
                        cv2.imshow(f'Frame', i)
                        cv2.waitKey(SPEED)

                if keys.ifpressed(pausekey):
                    time.sleep(.05)
                    if pause == True:
                        pause = False
                    else:
                        pause = True
                    time.sleep(0.05)
                if keys.ifpressed(playbackkey):
                    playback = True
                if keys.ifpressed(playforwardkey):
                    playforward = True
                if keys.ifpressed(fastkey):
                    fast = True
                if keys.ifpressed(slowkey):
                    slow = True
                if keys.ifpressed(prescisionkey):
                    if prescision == True:
                        prescision = False
                    else:
                        prescision = True
                    time.sleep(0.5)

                if pause == True:
                    if keys.ifpressed(markkey) and delayloop==loops and prev!=elementindex:

                        if mlist == []:
                            mlist.append(elementindex)

                        elif mlist != []:
                            mlist.append(elementindex)
                            markings.append(mlist)
                            mlist = []
                        print('MARKINGS')
                        print(mlist)
                        print(markings)
                        prev=elementindex

                    if playforward == True:
                        elementindex = elementindex + 1
                        if prescision == True:
                            time.sleep(0.5)
                    elif playback == True:
                        elementindex = elementindex - 1
                        if prescision == True:
                            time.sleep(0.5)
                    else:
                        elementindex = elementindex

                else:
                    elementindex = elementindex + 1
                    if fast == True:
                        if SPEED >= 2:
                            SPEED = SPEED - 1
                        elif SPEED > defSPEED:
                            SPEED = SPEED - 10
                    else:
                        if SPEED < defSPEED:
                            SPEED = defSPEED
                    if slow == True:
                        if SPEED < 200:
                            SPEED = SPEED + 10

                if delayloop == loops:
                    loops = 0

                if lenofdata == elementindex:
                    break

            if skip == True:
                break

            data = list(data)
            data2 = []
            nlist = [0]

            for mlist in markings:
                for i in mlist:
                    nlist.append(i)
            nlist.append(len(data) - 1)

            print(nlist)

            for i in range(len(nlist) - 1):
                if i % 2 == 0:
                    for n in data[nlist[i]:nlist[i + 1]]:
                        data2.append(n)

            for elementlist in data2:
                if keys.ifpressed(skipviewkey):
                    break
                if elementlist != 0:
                    for i in elementlist:

                        if len(i.shape) <= 1:
                            print(i)
                        else:
                            cv2.imshow(f'Frame', i)
                            cv2.waitKey(6)

            lenofdata = len(data2)

            while (True):

                command = input('SAVE ? [y/n] OR REDO ? [r]: ')
                if command == 'y':
                    print('SAVING...')
                    filepath = f'{savefilepath}\\cleaned-{lenofdata}-{tfBlitz.getdateandtime(time.time())}.npy'
                    with open(f'{savefilepath}\\cleanedinfo.txt', 'a') as f:
                        f.write(path + '\n')
                    np.save(filepath, data2)
                    print('SAVED')
                    break
                elif command == 'n':
                    print('unsaved')
                    break
                elif command == 'r':
                    redo = True
                    break
                else:
                    print('REENTER COMMAND')

            if redo == False:
                break






