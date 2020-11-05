from keras.models import load_model
import numpy as np
import csv


if __name__ == '__main__':

    MODEL_NAME = 'model/multimodal_model_acc_0.4738.h5'

    print('test1/test2/test3')
    tmp = input()

    TEXT_DATA = 'features/text_BN_' + tmp + '.npy'
    SPEECH_DATA = 'features/speech_' + tmp + '.npy'
    VIDEO_DATA = 'features/video_BN_' + tmp + '.npy'

    emotion_list = ['hap', 'ang', 'dis', 'fea', 'neu', 'sad', 'sur']

    model = load_model(MODEL_NAME)
    t = np.load(TEXT_DATA)
    print(t)
    v = np.load(VIDEO_DATA)
    print(v.shape)
    s = np.load(VIDEO_DATA)
    print(s)

    output = np.argmax(model.predict([t, v], verbose = 1 , batch_size = 256), axis = 1)

    results = []
    for i in range(len(output)):
        results.append(emotion_list[output[i]])

    print(results)
    i = 0
    with open('tt.csv', 'r') as f_2, open(tmp + '.csv', 'w') as f:
        csv_writer = csv.writer(f)
        data = csv.reader(f_2)
        csv_writer.writerow(["FileID","Emotion"])
        for line, row in zip(results,data):
                row.append(line)
                csv_writer.writerow(row)