from tensorflow import keras
import tensorflow as tf
import time
import cv2
import mss
import numpy
import directkeys
import keys
import tfBlitz
import pyvjoy
import collections
import win32api

j = pyvjoy.VJoyDevice(1)

vjoy_max = 32768


tfBlitz.setmemorylimit(True)
tfBlitz.setmemorylimit(4800)

que=[]
predictionss=[numpy.array([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]])]

filepath=r'E:\AI DATA\models\model-7'



SCREEN_CAPTURE_AREA={'top': 80, 'left': 80, 'width': 1280, 'height': 720}
OUTPUT_IMAGE_RESOLUTION=(400,300)

model0=keras.models.load_model(filepath)

model0.load_weights(r'E:\AI DATA\models\model-7 checkpoint\weights.01-1.69.hdf5')

with mss.mss() as sct:
    last_time=time.time()
    img_data=[]
    pos=[]
    images=[]
    predictions=[0,0,0,0,0,0,0]
    for i in range(1):
        print(i)
        time.sleep(1)
    frames=0
    with tf.device('/gpu:0'):
        while 'Screen capturing':#60 FPS AVERAGE
            lasttime=time.time()
            frames=frames+1
            img=sct.grab(SCREEN_CAPTURE_AREA)
            img=numpy.array(img)
            img=img[:,:,:1]
            img=cv2.resize(img,OUTPUT_IMAGE_RESOLUTION)

            # img=tf.reshape(img,(-1,300,400,1))
            img_data.append(img)
            if len(img_data)==2:
                cv2.imshow('win', img_data[1])
                cv2.waitKey(1)
                img_data=numpy.array(img_data)
                img_data=tf.reshape(img_data,(-1,2,300,400,1))
                predictionss=model0.predict(
                img_data, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=0, use_multiprocessing=True)
                tf.keras.backend.clear_session()

                img_data=list(img_data)
                img_data.pop(0)
            predictions=predictionss[0].astype(numpy.float32)
            predictions=predictions[0].astype(numpy.int32)
            print(predictions)

            # j.data.wAxisX = int((1-(predictions[1]+predictions[3]+predictions[4]+predictions[5])) * vjoy_max)
            # j.data.wAxisY = int((predictions[0]+predictions[4]+predictions[5]) * vjoy_max)
            # j.data.wAxisZ = int(predictions[2] * vjoy_max)


            # j.update()

            if keys.ifpressed('0'):
                time.sleep(0.5)
                directkeys.ReleaseKey(directkeys.A)
                directkeys.ReleaseKey(directkeys.S)
                directkeys.ReleaseKey(directkeys.D)
                directkeys.ReleaseKey(directkeys.W)
                while 1:

                    if keys.ifpressed('0'):
                        break
                    j.update()


            if numpy.array_equal(predictions,[1,0,0,0,0,0,0]) :
                directkeys.PressKey(directkeys.W)
                directkeys.ReleaseKey(directkeys.A)
                directkeys.ReleaseKey(directkeys.S)
                directkeys.ReleaseKey(directkeys.D)

            elif numpy.array_equal(predictions,[0,1,0,0,0,0,0]):
                directkeys.PressKey(directkeys.A)
                directkeys.ReleaseKey(directkeys.W)
                directkeys.ReleaseKey(directkeys.S)
                directkeys.ReleaseKey(directkeys.D)

            elif numpy.array_equal(predictions,[0,0,1,0,0,0,0]):
                directkeys.PressKey(directkeys.S)
                directkeys.ReleaseKey(directkeys.W)
                directkeys.ReleaseKey(directkeys.A)
                directkeys.ReleaseKey(directkeys.D)

            elif numpy.array_equal(predictions,[0,0,0,1,0,0,0]):
                directkeys.PressKey(directkeys.D)
                directkeys.ReleaseKey(directkeys.W)
                directkeys.ReleaseKey(directkeys.A)
                directkeys.ReleaseKey(directkeys.S)

            elif numpy.array_equal(predictions,[0,0,0,0,1,0,0]):
                directkeys.PressKey(directkeys.W)
                directkeys.PressKey(directkeys.A)
                directkeys.ReleaseKey(directkeys.S)
                directkeys.ReleaseKey(directkeys.D)

            elif numpy.array_equal(predictions,[0,0,0,0,0,1,0]):
                directkeys.PressKey(directkeys.W)
                directkeys.PressKey(directkeys.D)
                directkeys.ReleaseKey(directkeys.A)
                directkeys.ReleaseKey(directkeys.S)

            else:
                directkeys.ReleaseKey(directkeys.A)
                directkeys.ReleaseKey(directkeys.S)
                directkeys.ReleaseKey(directkeys.D)
                directkeys.ReleaseKey(directkeys.W)

            if frames==20:
                frames=0
                print(1/(time.time()-lasttime))

            print(predictions)
