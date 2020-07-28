import tensorflow as tf
import os
import glob
import numpy as np

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
# >>> uint8 and (800,600)



# tweaks={
#     'SHUFFLE_BUFFER_SIZE':1000,
#     'BATCH_SIZE':40,
#     'PREFETCH_NUM_OF_BATCHES':2 -->> prefetchs 2 batches into GPU memory which is 40x2 elements
# }

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
    print('----->>>  EXAMPLE PROTO  <<<-----')
    print(example_proto)
    features={}

    for line in tffile:
        info=line.split('_')
        features.update({'array'+info[0]:tf.io.FixedLenFeature([],tf.string)})

    parsed_features = tf.io.parse_single_example(example_proto, features)

    elements=[]

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
            print(f'UN-KNOWN DTYPE -- {dtype} -- FOUND')
            print('UPDATE THIS DTYPE IN tfBlitz.py under def parsefunction(example_proto)')

        array=tf.reshape(array,shape=shape)

        if parse_parms['normalize']==True:
            if dtype=='uint8':
                array=tf.cast(tf.divide(array,255),tf.float32)
            elif dtype=='int32':
                array=tf.cast(tf.divide(array,1),tf.float32)
            else:
                print(f'UN-KNOWN DTYPE -- {dtype} -- FOUND')
                print('UPDATE THIS DTYPE IN tfBlitz.py under def parsefunction(example_proto)')

        elements.append(array)
    returns=[]

    output=tf.stack([elements[1],elements[2]])

    returns.append(elements[0])
    returns.append(output)
    print('----->>>  SYMBOLIC TENSORS  <<<-----')
    for i in returns:
        print(i)

    returns=tuple(returns)
    tffile.clear()

    return returns

def generatetfrecords(datafilepath,tfrecordsfilepath):
    firstloop = True
    info=[]

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
        data=os.path.basename(path)

        tfrecordwriter = tf.io.TFRecordWriter(f'{tfrecordsfilepath}\\{data[:-4]}.tfrecord')

        data = np.load(path, allow_pickle=True)


        for i in data:

            features = {}

            count = 0
            for array in i:
                shape=array.shape
                dtype=array.dtype
                if firstloop==True:
                    info.append(f'{count}_{shape}_{dtype}')
                features.update({f'array{count}': _bytes_feature(array.tostring())})
                count = count + 1

            if firstloop==True:
                with open(f'{tfrecordsfilepath}\\tfDATAinfo.txt', 'w') as f:
                    for lines in info:
                        f.write(lines+'\n')
                    firstloop=False

            example=tf.train.Example(features=tf.train.Features(feature=features))
            tfrecordwriter.write(example.SerializeToString())
        with open(f'{tfrecordsfilepath}\\tfPATHinfo.txt', 'a') as f:
            f.write(path + '\n')

        tfrecordwriter.close()
        filecount=filecount-1
        print(str(filecount)+' FILES_LEFT ')
    return

def dataset(tfrecordsfilepath,tweaks=None,normalize=False):

    parse_parms.update({'normalize':normalize})

    tfpaths=glob.glob(tfrecordsfilepath+'\\*.tfrecord')
    txtpath=tfrecordsfilepath+'\\tfDATAinfo.txt'

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

