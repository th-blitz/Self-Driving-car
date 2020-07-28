import time
import cv2
import mss
import numpy
import keys
import mytime

SCREEN_CAPTURE_AREA={'top': 80, 'left': 80, 'width': 1280, 'height': 720}
OUTPUT_IMAGE_RESOLUTION=(400,300)
MASTER_FOLDER_DIRECTORY=r'E:\AI DATA\DATA'
START_or_PAUSE_RECORDING_KEY='spacebar'
KEYS_TO_RECORD=['w','a','s','d']
AUTOSAVE_AFTER_n_FRAMES=2000
SHOW_RECORDING_WINDOW=False
RECORD_AT=1
FILE_NAME='data'
ENABLE_AUTO_SAVE=True
PLAY_FORWARD_KEY='l'
PLAY_BACK_KEY='k'

command=None
pause=False

with mss.mss() as sct:
    last_time=time.time()
    img_data=[]
    pos=[]

    for i in range(4):
        print(i)
        time.sleep(1)
    frames=0

    while 'Screen capturing':#60 FPS AVERAGE
        frames=frames+1
        img=sct.grab(SCREEN_CAPTURE_AREA)
        img=numpy.array(img)
        img=img[:,:,:1]
        img=cv2.resize(img,OUTPUT_IMAGE_RESOLUTION)
        keystrokes=keys.recordKeyStrokes(KEYS_TO_RECORD)
        # pos=wapi.GetCursorPos()
        keystrokes=numpy.array(keystrokes)
        img_data.append([img,keystrokes])

        if keys.ifpressed(START_or_PAUSE_RECORDING_KEY):
            time.sleep(0.05)
            if pause==False:
                pause=True
            else:
                pause=False
            time.sleep(0.05)
        cv2.waitKey(RECORD_AT)
        if SHOW_RECORDING_WINDOW==True:
            cv2.imshow('window',img)
            if cv2.waitKey(RECORD_AT) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        if frames==120:
            frames=0
            fps=1/(time.time()-last_time)
            fps=numpy.array(fps)
            fps=fps.astype(numpy.float16)
            print(f'fps: {fps}')
        last_time = time.time()
        if (len(img_data)==(AUTOSAVE_AFTER_n_FRAMES) or
                pause==True):

            print('TOTAL FRAMES RECOEDED')
            print(len(img_data))

            if not pause==True:

                while(True):
                    if ENABLE_AUTO_SAVE==False:
                        command = input('SAVE DATA ? [y/n] : ')
                    else:
                        pause=False
                        command='y'
                    if command=='y':
                        print('SAVING.....')
                        dateandtime=mytime.getdateandtime(time.time())
                        datafile=[]
                        img_data2=[]
                        count=0

                        try:
                            with open(f'{MASTER_FOLDER_DIRECTORY}\\datainfo.txt', 'r') as f:
                                for line in f:
                                    stripped_line = line.strip()
                                    datafile.append(stripped_line)
                                    count=int(datafile[-1])+1
                        except:
                            with open(f'{MASTER_FOLDER_DIRECTORY}\\datainfo.txt', 'w') as f:
                                f.write(str(count)+'\n')
                        filepath = f'{MASTER_FOLDER_DIRECTORY}\\{count}-Res-{OUTPUT_IMAGE_RESOLUTION}-len-{len(img_data)}-{dateandtime}.npy'
                        with open(f'{MASTER_FOLDER_DIRECTORY}\\datainfo.txt', 'a') as f:
                            f.write(str(count) + '\n')
                        numpy.save(filepath, img_data)
                        print('.DATA SAVED.')
                        img_data.clear()
                        break
                    elif command=='n':
                        print('DUMPING COLLECTED DATA.....')
                        img_data.clear()
                        break
                    else:
                        print('CAREFULL , ENTER THE RIGHT COMMAND AGAIN')
                        print('ELSE YOU WILL RISK LOSING YOUR COLLECTED DATA')

            if pause==True:
                print('PAUSED')
                lenofdata=len(img_data)
                elementindex=lenofdata-1
                prev=0
                while(True):
                    cv2.imshow('window',img_data[elementindex][0])
                    cv2.waitKey(25)
                    if elementindex!=prev:
                        print(elementindex)
                        prev=elementindex
                    if elementindex<lenofdata-1:
                        if keys.ifpressed(PLAY_FORWARD_KEY):
                            elementindex = elementindex + 1
                    if keys.ifpressed(PLAY_BACK_KEY):
                        elementindex = elementindex - 1
                    if keys.ifpressed(START_or_PAUSE_RECORDING_KEY):
                        print('UN-PAUSED')
                        # pause=False
                        img_data=img_data[:elementindex]
                        break

        if command=='e':
            break
