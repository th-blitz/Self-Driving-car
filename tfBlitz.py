import tensorflow as tf
import os
import glob
import numpy as np
from random import shuffle
import datetime
import time
from contextlib import redirect_stdout
from tensorflow import keras

# use single folder directory for a single data type
# only one tfinfo.txt file is created for a single data type in a single folder directory
# tfinfo.txt contains valuable information of the data type used for unpacking .tfrecord files
# NEVER DELETE tfinfo.txt else data won't be unpacked
# In case if tfinfo.txt was deleted enter data type of each element into the below list
# format 'nth-element_shape_dtype'

tffile=[]
tweaksenabled=[]
parse_parms={}

# Eg: for 2 elements X , Y of shapes (800,600), (4,) and dtypes uint8, int32
# THEN tffile=['0_(800,600)_uint8','1_(4,)_int32']
# Eg: for n elements A,B,C...etc of shapes (xxx,yyy),(xx,yy),(aaa,fff)...etc and dtypes float32,int64,int32... etc
# THEN tffile=['0_(xxx,yyy)_float32','1_(xx,yy)_int64','2_(aaa,fff)_int32', . . . etc upto nth element]
# Incase if you dont know how to find dtype or shape use print(X.dtype) , print(X.shape) for nparrays




# tweaks={
#     'SHUFFLE_BUFFER_SIZE':1000,
#     'BATCH_SIZE':40,
#     'PREFETCH_NUM_OF_BATCHES':2 -->> prefetchs 2 batches into GPU memory which is 40x2 elements
# }

def tfwrite(filepath,data,firstloop):
    tfrecordwriter = tf.io.TFRecordWriter(f'{filepath}.tfrecord')
    info=[]
    for i in data:

        features = {}

        count = 0
        for array in i:
            shape = array.shape
            dtype = array.dtype
            if firstloop == True:
                info.append(f'{count}_{shape}_{dtype}')
            features.update({f'array{count}': _bytes_feature(array.tostring())})
            count = count + 1

        example = tf.train.Example(features=tf.train.Features(feature=features))
        tfrecordwriter.write(example.SerializeToString())

        if firstloop == True:
            with open(f'{os.path.dirname(filepath)}\\tfDATAinfo.txt', 'w') as f:
                for lines in info:
                    f.write(lines + '\n')
            firstloop=False

    tfrecordwriter.close()
    return

def listdevices(deviceslist):
    info=[]
    for device in deviceslist:
        info.append(tf.config.experimental.list_physical_devices(device))
    print(info)
    return info

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parsefunction(example_proto):
    print('==============================================================================')
    print('----------> >>  EXAMPLE DATA FORMAT USED  << <----------')
    print(example_proto)
    print('')
    features={}

    for line in tffile:
        info=line.split('_')
        features.update({'array'+info[0]:tf.io.FixedLenFeature([],tf.string)})

    parsed_features = tf.io.parse_single_example(example_proto, features)

    elements=[]

    print('----------> >>  EXPECTED DATA TO BE FED INTO THE PIPELINE  << <----------')

    for line in tffile:
        shapelist = []
        info=line.split('_')
        print(info)
        array=parsed_features['array'+info[0]]
        shape=eval(info[1])
        for i in shape:
            shapelist.append(i)

        # shapelist.insert(0,-1)
        shapelist.append(1)
        shape=tuple(shapelist)
        dtype=info[2]
        if dtype=='uint8':
            array=tf.io.decode_raw(array,tf.uint8)
        elif dtype=='int32':
            array=tf.io.decode_raw(array,tf.int32)
        elif dtype=='int64':
            array=tf.io.decode_raw(array,tf.int64)
        elif dtype=='float32':
            array=tf.io.decode_raw(array,tf.float32)
        elif dtype=='float64':
            array=tf.io.decode_raw(array,tf.float64)
        elif dtype=='complex64':
            array=tf.io.decode_raw(array,tf.complex64)
        elif dtype=='complex128':
            array=tf.io.decode_raw(array,tf.complex128)
        else:
            print(f'UN-KNOWN DTYPE {dtype} FOUND')
            print('UPDATE THIS DTYPE IN tfBlitz.py under def parsefunction(example_proto) UNDER DTYPE FUNCTIONS')

        array=tf.reshape(array,shape=shape)
#######################################################################################################
########################## TF.FLOAT32 IS BETTER THAN TF.FLOAT16 #######################################

        if parse_parms['normalize']==True:
            if dtype=='uint8':
                array=tf.cast(tf.divide(array,255),tf.float32)
            elif dtype=='int32':
                array=tf.cast(tf.divide(array,1),tf.float32)
            else:
                print(f'UN-KNOWN DTYPE -- {dtype} -- FOUND')
                print('UPDATE THIS DTYPE IN tfBlitz.py under def parsefunction(example_proto) UNDER NORMALIZATION FUNCTIONS')

#######################################################################################################

        elements.append(array)
    returns=[]

    x=parse_parms['X']
    y=parse_parms['Y']

    print('')
    print('----------> >>  DATA TO STACK  << <----------')
    print(f'INPUT OR X = STACK DATA INDEXES {x}')
    print(f'OUTPUT OR Y = STACK DATA INDEXES {y}')
    print('')

    if x==(None) and y==(None):
        for i in elements:
            returns.append(i)

    elif isinstance(x,int) and isinstance(y,int):
        returns.append(elements[x])
        returns.append(elements[y])

    elif isinstance(x,int) and len(y)>1:
        returns.append(elements[x])
        stack=[]
        for i in y:
            stack.append(elements[i])
        returns.append(tf.stack(stack))

    elif len(x)>1 and isinstance(y,int):
        stack=[]
        for i in x:
            stack.append(elements[i])
        returns.append(tf.stack(stack))
        returns.append(elements[y])

    elif len(x)>1 and len(y)>1:
        stack=[]
        for i in x:
            stack.append(elements[i])
        returns.append(tf.stack(stack))
        stack=[]
        for i in y:
            stack.append(elements[i])
        returns.append(tf.stack(stack))

    else:
        print('ERROR AT STACK FUNCTIONS IN TFBLITZ.PY DEF PARSE FUNCTION')

    # output=tf.stack([elements[1],elements[2]])
    # output=elements[1]
    # returns.append(elements[0])
    # returns.append(output)

    print('----------> >>  FINAL DATA TO COME OUT OF THE PIPELINE  << <----------')
    for i in returns:
        print(i)
    print('')

    returns=tuple(returns)
    tffile.clear()

    return returns

def generatetfrecords(datafilepath,tfrecordsfilepath):
    firstloop = True

    paths=glob.glob(f'{datafilepath}\\*.npy')
    try:

        with open(f'{tfrecordsfilepath}\\tfPATHinfo.txt', 'r') as f:
            for line in f:
                stripped_line = line.strip()
                try:
                    paths.remove(stripped_line)
                except:
                    pass

    except:
        pass
    filecount=len(paths)
    print('FILES_LEFT : '+ str(filecount))

    for path in paths:
        dataname = os.path.basename(path)
        dataname=dataname[:-4]
        data = np.load(path, allow_pickle=True)
        filepath=f'{tfrecordsfilepath}\\{dataname}'

        tfwrite(filepath,data,firstloop)


        if firstloop == True:
                firstloop = False


        with open(f'{tfrecordsfilepath}\\tfPATHinfo.txt', 'a') as f:
            f.write(path + '\n')

        filecount=filecount-1
        print(str(filecount)+' FILES_LEFT ')
    return

def dataset(tfrecordsfilepath,tweaks=None,normalize=False,stack_X=(None),stack_Y=(None)):

    parse_parms.update({'normalize':normalize})
    parse_parms.update({'X':stack_X})
    parse_parms.update({'Y':stack_Y})

    if type(tfrecordsfilepath) is str:
        tfpaths=glob.glob(tfrecordsfilepath+'\\*.tfrecord')
        txtpath = tfrecordsfilepath + '\\tfDATAinfo.txt'

    elif type(tfrecordsfilepath) is list:
        tfpaths=tfrecordsfilepath
        filepath=os.path.dirname(tfpaths[0])
        txtpath=filepath + '\\tfDATAinfo.txt'
    else:
        print('TF-FILE-PATHS-ERROR')
        txtpath=None
        tfpaths=None

    with open(txtpath, "r") as a_file:
        for line in a_file:
            stripped_line = line.strip()
            tffile.append(stripped_line)

    mytffiles = tf.data.Dataset.list_files(tfpaths)
    dataset = tf.data.TFRecordDataset(mytffiles)

    if tweaks!=None:
        dataset = dataset.shuffle(tweaks['SHUFFLE_BUFFER_SIZE'],
                                  reshuffle_each_iteration=True)
    dataset = dataset.map(parsefunction)
    if tweaks!=None:
        dataset=dataset.batch(tweaks['BATCH_SIZE'],drop_remainder=True)
        dataset=dataset.prefetch(tweaks['PREFETCH_NUM_OF_BATCHES'])

    return dataset

def setmemorygrowth(bool=True):
    gpus=tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, bool)
        except RuntimeError as e:
            print(e)

def setmemorylimit(memory=6144):
    gpus=tf.config.experimental.list_physical_devices('GPU')
    if gpus:

        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu, [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
        except RuntimeError as e:
            print(e)

def train_test_datasets(tfrecordsfilepath,tweaks=None,normalize=False,validation_files_split=0.05,stack_X=(0),stack_Y=(1)):

    tfpaths=glob.glob(f'{tfrecordsfilepath}\\*.tfrecord')

    shuffle(tfpaths)
    shuffle(tfpaths)

    numoffiles=len(tfpaths)
    splitindex=int(numoffiles*validation_files_split)

    val_tffiles=tfpaths[:splitindex]
    train_tffiles=tfpaths[splitindex:]

    print(val_tffiles)
    print(train_tffiles)

    train_dataset=dataset(train_tffiles,tweaks,normalize,stack_X,stack_Y)
    test_dataset=dataset(val_tffiles,tweaks,normalize,stack_X,stack_Y)

    return train_dataset,test_dataset

def getdateandtime(timestamp_in_sec=None):
    if timestamp_in_sec==None:
        timestamp_in_sec=time.time()
    dateandtime = str(datetime.datetime.fromtimestamp(timestamp_in_sec))
    dateandtime = list(dateandtime)
    dateandtime[10] = '_'
    dateandtime[13] = '-'
    dateandtime[16] = '-'
    dateandtime[19] = '_'

    string = ''
    for i in dateandtime:
        string = string + i
    return string

def getpaths(from_directory,to_directory,filetype0='npy',filetype1='tfrecord'):
    frompaths=glob.glob(f'{from_directory}\\*.{filetype0}')
    topaths=glob.glob(f'{to_directory}\\*.{filetype1}')

    from_names=[]
    to_names=[]

    for path in frompaths:
        name=os.path.basename(path)
        name,filetype=name.split('.')
        from_names.append(name)

    for path in topaths:
        name = os.path.basename(path)
        name, filetype = name.split('.')
        to_names.append(name)

    for name in to_names:
        try:
            from_names.remove(name)
        except:
            pass

    from_paths=[]

    for name in from_names:
        from_paths.append(f'{from_directory}\\{name}.{filetype0}')

    return from_paths

def Save(model,filepath,additional_info=None):

    model.save(
        filepath, overwrite=True, include_optimizer=True, save_format=None,
        signatures=None, options=None
    )

    with open(f'{filepath}\\modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

        if additional_info!=None:
            f.write(str(additional_info))

    return
