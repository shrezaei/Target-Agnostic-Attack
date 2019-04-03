from keras.preprocessing import image
from keras_vggface import utils
import os, pdb
import numpy as np

BalancedTrainingSet = True

def loadData(dirPath):
    img_list = []
    for folder, subs, files in os.walk(dirPath):
        for file in files:
            filename = folder + "/" + file
            img = image.load_img(filename, target_size=(224, 224))
            img = image.img_to_array(img)
            img = utils.preprocess_input(img, version=1)
            img_list.append(img)
    data = np.array(img_list)
    if BalancedTrainingSet:
        trainData = data[:35] #For undersampled case where training data is even among classes
    else:
        trainData = data[:-5]
    testData = data[:-5]

    return trainData, testData


data0, testdata0 = loadData("data/retrain/George_W_Bush/")
data1, testdata1 = loadData("data/retrain/Colin_Powell/")
data2, testdata2 = loadData("data/retrain/Tony_Blair/")
data3, testdata3 = loadData("data/retrain/Donald_Rumsfeld/")
data4, testdata4 = loadData("data/retrain/Gerhard_Schroeder/")
data5, testdata5 = loadData("data/retrain/Ariel_Sharon/")
data6, testdata6 = loadData("data/retrain/Hugo_Chavez/")
data7, testdata7 = loadData("data/retrain/Jean_Chretien/")
data8, testdata8 = loadData("data/retrain/John_Ashcroft/")
data9, testdata9 = loadData("data/retrain/Junichiro_Koizumi/")
data10, testdata10 = loadData("data/retrain/Serena_Williams/")
data11, testdata11 = loadData("data/retrain/Jacques_Chirac/")
data12, testdata12 = loadData("data/retrain/Vladimir_Putin/")
data13, testdata13 = loadData("data/retrain/Luiz_Inacio_Lula_da_Silva/")
data14, testdata14 = loadData("data/retrain/Gloria_Macapagal_Arroyo/")
data15, testdata15 = loadData("data/retrain/Jennifer_Capriati/")
data16, testdata16 = loadData("data/retrain/Arnold_Schwarzenegger/")
data17, testdata17 = loadData("data/retrain/Lleyton_Hewitt/")
data18, testdata18 = loadData("data/retrain/Laura_Bush/")
data19, testdata19 = loadData("data/retrain/Alejandro_Toledo/")

label0 = np.ones(data0.shape[0]) * 1
label1 = np.ones(data1.shape[0]) * 2
label2 = np.ones(data2.shape[0]) * 3
label3 = np.ones(data3.shape[0]) * 4
label4 = np.ones(data4.shape[0]) * 5
label5 = np.ones(data5.shape[0]) * 6
label6 = np.ones(data6.shape[0]) * 7
label7 = np.ones(data7.shape[0]) * 8
label8 = np.ones(data8.shape[0]) * 9
label9 = np.ones(data9.shape[0]) * 10
label10 = np.ones(data10.shape[0]) * 11
label11 = np.ones(data11.shape[0]) * 12
label12 = np.ones(data12.shape[0]) * 13
label13 = np.ones(data13.shape[0]) * 14
label14 = np.ones(data14.shape[0]) * 15
label15 = np.ones(data15.shape[0]) * 16
label16 = np.ones(data16.shape[0]) * 17
label17 = np.ones(data17.shape[0]) * 18
label18 = np.ones(data18.shape[0]) * 19
label19 = np.ones(data19.shape[0]) * 20


data = np.concatenate((data0, data1, data2, data3, data4, data5, data6, data7, data8, data9,
                       data10, data11, data12, data13, data14, data15, data16, data17, data18, data19), axis=0)
label = np.concatenate((label0, label1, label2, label3, label4, label5, label6, label7, label8, label9,
                        label10, label11, label12, label13, label14, label15, label16, label17, label18, label19), axis=0)


perm = np.arange(data.shape[0])
np.random.shuffle(perm)
data = data[perm]
label = label[perm]
np.save('trainData.npy', data)
np.save('trainLabel.npy', label)


label0 = np.ones(testdata0.shape[0]) * 1
label1 = np.ones(testdata1.shape[0]) * 2
label2 = np.ones(testdata2.shape[0]) * 3
label3 = np.ones(testdata3.shape[0]) * 4
label4 = np.ones(testdata4.shape[0]) * 5
label5 = np.ones(testdata5.shape[0]) * 6
label6 = np.ones(testdata6.shape[0]) * 7
label7 = np.ones(testdata7.shape[0]) * 8
label8 = np.ones(testdata8.shape[0]) * 9
label9 = np.ones(testdata9.shape[0]) * 10
label10 = np.ones(testdata10.shape[0]) * 11
label11 = np.ones(testdata11.shape[0]) * 12
label12 = np.ones(testdata12.shape[0]) * 13
label13 = np.ones(testdata13.shape[0]) * 14
label14 = np.ones(testdata14.shape[0]) * 15
label15 = np.ones(testdata15.shape[0]) * 16
label16 = np.ones(testdata16.shape[0]) * 17
label17 = np.ones(testdata17.shape[0]) * 18
label18 = np.ones(testdata18.shape[0]) * 19
label19 = np.ones(testdata19.shape[0]) * 20

testdata = np.concatenate((testdata0, testdata1, testdata2, testdata3, testdata4, testdata5, testdata6, testdata7, testdata8, testdata9,
                       testdata10, testdata11, testdata12, testdata13, testdata14, testdata15, testdata16, testdata17, testdata18, testdata19), axis=0)
testlabel = np.concatenate((label0, label1, label2, label3, label4, label5, label6, label7, label8, label9,
                        label10, label11, label12, label13, label14, label15, label16, label17, label18, label19), axis=0)

np.save('testData.npy', testdata)
np.save('testLabel.npy', testlabel)
