from tensorflow import keras
import tfBlitz
import datetime

def train_model(INPUTSHAPE,FILEPATH,TRAINING_DATA_FILEPATH,INPUTBATCH):
    
    
    def leaky_relu(alpha=0.3):
        return keras.layers.LeakyReLU(alpha=alpha)
    
    CHECKPOINT=FILEPATH+r'\weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    # INPUTSHAPE=(300,400,1)
    # INPUTBATCH=40
    
    tweaks={
        'SHUFFLE_BUFFER_SIZE':1000,
        'BATCH_SIZE':INPUTBATCH,
        'PREFETCH_NUM_OF_BATCHES':2
    }
    
    
    tfBlitz.setmemorygrowth(True)
    tfBlitz.setmemorylimit(4800)
    
    
    input0=keras.Input(
        shape=INPUTSHAPE
    )
    
    chain0=keras.layers.Convolution2D(32,kernel_size=(8,8),strides=(1,1),padding='valid',activation='relu')(input0)
    chain0=keras.layers.Convolution2D(32,kernel_size=(8,8),strides=(1,1),padding='valid',activation='relu')(chain0)
    chain0=keras.layers.MaxPooling2D(pool_size=(2,2),strides=None)(chain0)
    chain0=keras.layers.Convolution2D(64,kernel_size=(6,6),strides=(1,1),padding='valid',activation='relu')(chain0)
    chain0=keras.layers.Convolution2D(64,kernel_size=(6,6),strides=(1,1),padding='valid',activation='relu')(chain0)
    chain0=keras.layers.MaxPooling2D(pool_size=(2,2))(chain0)
    chain0=keras.layers.Convolution2D(128,kernel_size=(3,3),strides=(1,1),padding='valid',activation=leaky_relu())(chain0)
    chain0=keras.layers.Convolution2D(128,kernel_size=(3,3),strides=(1,1),padding='valid',activation=leaky_relu())(chain0)
    chain0=keras.layers.MaxPooling2D(pool_size=(4,4))(chain0)
    # chain0=keras.layers.Convolution2D(128,kernel_size=(3,3),strides=(1,1),padding='valid',activation=leaky_relu())(chain0)
    # chain0=keras.layers.Convolution2D(128,kernel_size=(3,3),strides=(1,1),padding='valid',activation=leaky_relu())(chain0)
    # chain0=keras.layers.MaxPooling2D(pool_size=(2,2))(chain0)
    
    chain0=keras.layers.Flatten()(chain0)
    chain0=keras.layers.Dense(512,activation=leaky_relu())(chain0)
    chain0=keras.layers.Dropout(0.05)(chain0)
    
    chain0=keras.layers.Dense(256,activation=leaky_relu())(chain0)
    chain0=keras.layers.Dropout(0.04)(chain0)
    output=keras.layers.Dense(9,activation='softmax')(chain0)
    
    model0=keras.models.Model(inputs=input0,outputs=output)
    model0.summary()
    
    
    log_dir = "logs\\with_profiler\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,profile_batch=2,update_freq=10)
    
    checkpont_callback=keras.callbacks.ModelCheckpoint(
        CHECKPOINT, monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=False, mode='auto', save_freq='epoch'
    )
    
    optimizer=keras.optimizers.Adam(learning_rate=1e-5)
    
    model0.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    
    tfBlitz.Save(model0,FILEPATH,additional_info='lr = 1e-5')
    
    train_dataset,test_dataset=tfBlitz.train_test_datasets(
        TRAINING_DATA_FILEPATH,tweaks=tweaks,normalize=True,validation_files_split=0.04,stack_X=(0),stack_Y=(1)
    )
    model0.fit(train_dataset,validation_data=test_dataset,epochs=3,callbacks=[tensorboard_callback,checkpont_callback])
    
