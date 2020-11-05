import os
import numpy as np
import cv2
import csv
import time

import tensorflow as tf

from mtcnn import FaceCropperAll
from preprocessing import Preprocessor
from keras_vggface.vggface import VGGFace
from keras import Model

# set gpu-memory usages
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# load models
facecropper = FaceCropperAll(type='mtcnn', resizeFactor=1.0)
preprocessor = Preprocessor()
model = VGGFace(input_shape=(224,224,3), include_top=True, weights='vggface')
vggface = Model(inputs=model.input, outputs=model.get_layer('fc6').output)

def test_input(parentPath, interval=10):

    frame_seq = os.listdir(parentPath)
    frame_seq.sort()

    input_list = []

    max_second = 10
    cnt = 0

    if(len(frame_seq) < max_second * interval):
        N = len(frame_seq)
    else:
        N = max_second * interval

    for i in range(0, N, interval):
        frame_seqAbsPath = os.path.join(parentPath, frame_seq[i])
        try:
            img = cv2.imread(frame_seqAbsPath)
            input_list.append(img)
        except Exception as ex:
            print("Error:", ex)

    return input_list

def testFaceCropper(input_list):
    global facecropper
    res, cnt = facecropper.detectMulti(input_list)
    if(res is None):
        print('TEST FAIL FACE CROPPER')
        return None
    return res, cnt


def pre_processing(input_list):
    global preprocessor
    res = preprocessor.process(input_list)
    if(res is None):
        print('TEST FAIL PREPROCESSOR')
        return None
    return res


def vgg_face(input_list):
    global vggface
    res = vggface.predict(input_list)
    if(res is None):
        print('TEST FAIL VGGFACE')
        return None
    return res


def feature_extract(frame_path):

    interval = 1  # Use 1 frame for 1 second (1 frame/sec)
    img = test_input(frame_path, interval)
    # face cropper
    resCropped, cnt = testFaceCropper(img)
    print("Face cropped:", cnt)

    # not detected
    if cnt == 0:
        return np.zeros((1, 4096))

    # pre-processing
    resPreprop = pre_processing(resCropped)

    # VGGface feature
    video_feature = vgg_face([resPreprop])
    return video_feature


if __name__ == '__main__':
    savePath = 'features/val/'
    video_file_path =  sorted(glob.glob('../../dataset/qia2020/image_frame_test/*'))
    for i in range(len(video_file_path)):
        img_path = video_file_path[i]
        # print(video_file_path[i].split("/")[-1])
        save_path = savePath + video_file_path[i].split("/")[-1] + '.npy'
        #print(save_path)
        #print(video_file_path[i])
        if(os.path.isfile(save_path)):
            print("already done:", i)
        else:
            print("data number:", i)
            features = feature_extract(frame_path=img_path)
            np.save(save_path, features)
                                                