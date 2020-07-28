from tensorflow import keras
import tfBlitz
import datetime
import tensorflow as tf

def traindataset():

    while True:
        traindataset = tfBlitz.dataset(r'E:\AI DATA\tffiles\timedistributed', tweaks,normalize=True)
        for x,y in traindataset:

            yield (x,y)


def valdataset():


    while True:
        valdataset = tfBlitz.dataset(r'E:\AI DATA\tffiles\timedistributed\val',tweaks,normalize=True)
        for u,v in valdataset:

            yield (u,v)


CHECKPOINT=r'E:\AI DATA\models\model-9 checkpoint\weights.{epoch:02d}-{val_loss:.2f}.hdf5'
FILEPATH=r'E:\AI DATA\models\model-9'
INPUTSHAPE0=(2,300,400,1)
INPUTSHAPE1=(7,1)
INPUTBATCH=16

tweaks={
    'SHUFFLE_BUFFER_SIZE':640,
    'BATCH_SIZE':INPUTBATCH,
    'PREFETCH_NUM_OF_BATCHES':1
}


tfBlitz.setmemorygrowth(True)
tfBlitz.setmemorylimit(4800)


input0=keras.Input(
    shape=INPUTSHAPE0
)

chain0=keras.layers.TimeDistributed(keras.layers.Convolution2D(32,kernel_size=(8,8),strides=(1,1),padding='valid'))(input0)
chain0=keras.layers.TimeDistributed(keras.layers.ReLU())(chain0)
chain0=keras.layers.TimeDistributed(keras.layers.Convolution2D(32,kernel_size=(6,6),strides=(1,1),padding='valid'))(chain0)
chain0=keras.layers.TimeDistributed(keras.layers.ReLU())(chain0)
chain0=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2,2),strides=None))(chain0)
chain0=keras.layers.TimeDistributed(keras.layers.Convolution2D(32,kernel_size=(3,3),strides=(1,1),padding='valid'))(chain0)
chain0=keras.layers.TimeDistributed(keras.layers.LeakyReLU(alpha=0.3))(chain0)
chain0=keras.layers.TimeDistributed(keras.layers.Convolution2D(32,kernel_size=(3,3),strides=(1,1),padding='valid'))(chain0)
chain0=keras.layers.TimeDistributed(keras.layers.LeakyReLU(alpha=0.3))(chain0)
chain0=keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(4,4)))(chain0)

chain0=keras.layers.TimeDistributed(keras.layers.Flatten())(chain0)

chain0=keras.layers.LSTM(256,activation='tanh',return_sequences=True)(chain0)

chain2=keras.layers.Dense(128)(chain0)
chain2=keras.layers.ReLU()(chain2)
chain2=keras.layers.Dropout(0.5)(chain2)
chain2=keras.layers.Dense(64)(chain2)
chain2=keras.layers.ReLU()(chain2)
chain2=keras.layers.Dropout(0.5)(chain2)
chain2=keras.layers.Dense(7)(chain2)
output=keras.layers.Activation('softmax')(chain2)




model0=keras.models.Model(inputs=input0,outputs=output)
model0.summary()

log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpont_callback=keras.callbacks.ModelCheckpoint(
    CHECKPOINT, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch'
)

optimizer=keras.optimizers.RMSprop(learning_rate=1e-3)


model0.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

model0.fit(traindataset(),validation_data=valdataset(),steps_per_epoch=1794,validation_steps=107,epochs=12,callbacks=[tensorboard_callback,checkpont_callback])

model0.save(
    FILEPATH, overwrite=True, include_optimizer=True, save_format=None,
    signatures=None, options=None
)
