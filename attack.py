from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Activation
from keras.models import load_model
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import sys

RetrainAllLayers = False
newLayers = 1

# np.random.seed(int(sys.argv[1]))
#newLayers = int(sys.argv[2])

# To indicate the last layer of feature extractor
retrainLayerIndex = -4

retrained_model_path = 'vggface-retrained.h5'


maxAttempt = 4095
confidenceLevel = 0.99
nb_class = 5
alpha = 0.1
epoch = 50
initImage = 2   #0:random; 1:a face image; other values:blank

VGGFace_original_model = VGGFace(model='vgg16', pooling='max')
vgg_retrained = load_model(retrained_model_path)


XX = VGGFace_original_model.input
YY = VGGFace_original_model.layers[retrainLayerIndex].output
feature_extractor = Model(XX, YY)


def generate_image(input, model, intermediate_target):
    loss = K.sum(K.square(model.output[0] - intermediate_target[0]))
    gradients = K.gradients(loss, model.input)[0]
    fn = K.function([model.input], [gradients])

    for i in range(1, epoch):
        grads = fn([input])
        input = input - alpha * grads[0]
    return input

def predictor():
    model = Sequential()
    model.add(Flatten(name='test', input_shape=(1,4096)))
    model.add(Activation('relu', name='fc7/relu'))
    if newLayers==1:
        model.add(Dense(nb_class, name='fc8-2'))
        model.add(Activation('softmax', name='fc8-2/softmax'))
    elif newLayers==2:
        model.add(Dense(4096, name='fc8-2'))
        model.add(Activation('relu', name='fc8-2/relu'))
        model.add(Dense(nb_class, name='fc8-2-1'))
        model.add(Activation('softmax', name='fc8-2/softmax'))
    elif newLayers==3:
        model.add(Dense(4096, name='fc8-2'))
        model.add(Activation('relu', name='fc8-2/relu'))
        model.add(Dense(4096, name='fc8-2-1'))
        model.add(Activation('relu', name='fc8-2-1/relu'))
        model.add(Dense(nb_class, name='fc8-2-2'))
        model.add(Activation('softmax', name='fc8-2/softmax'))
    return model

#This is the classifier part alone. It is not neccessary
#for the target-agnostic algorithm. It was used to track the progress of the code.
predictor_model = predictor()
predictor_model.load_weights(retrained_model_path, by_name=True)


x = np.zeros(shape=(4096,1,4096))
for i in range(0,4095):
    x[i,0,i] = 100000

preds = predictor_model.predict(x)

outs = []
outs2 = []

TargetLabel = np.ones(4095) * -1
AdvLabel = np.ones(4095) * -1
InitialLabel = np.ones(4095) * -1
flags = np.zeros(nb_class)

if initImage == 0:
    initial_image = np.random.rand(224, 224, 3) * 100
    initial_image = image.img_to_array(initial_image)
    initial_image = np.expand_dims(initial_image, axis=0)
elif initImage == 1:
    initial_image = image.load_img('student2/train/Tony_Blair/Tony_Blair_0008.jpg', target_size=(224, 224))
    initial_image = image.img_to_array(initial_image)
    initial_image = np.expand_dims(initial_image, axis=0)
else:
    initial_image = np.ones([1, 224, 224, 3]) * 100

initial_image = utils.preprocess_input(initial_image, version=1)    #Maybe only needed for blank image

failedImage95 = 0 #The number of generated images that didn't trigger any class with 95% confidence
failedImage99 = 0 #The number of generated images that didn't trigger any class with 99% confidence
totalAttemp = 0
FinalTotalAttempt=0
for j in range(0, maxAttempt):
#     i = 4094 - j
    i = j
    # i = np.random.randint(0,4094)
    outs.append(np.argmax(preds[i]))

    img = generate_image(initial_image, feature_extractor, x[i])
    output0 = vgg_retrained.predict(initial_image)
    confidence0 = np.max(output0)
    label0 = np.argmax(output0)
    output = vgg_retrained.predict(img)
    confidence = np.max(output)
    label = np.argmax(output)

    # np.argmax(preds[i]): What we aim for
    # label0: what is the original image class
    # label: what is the label of modified image
    print(output)
    print(i, np.argmax(preds[i]), label0, confidence0, label, confidence)
    print(flags)
    print(failedImage95, failedImage99)

    TargetLabel[i] = np.argmax(preds[i])
    AdvLabel[i] = label

    InitialLabel[i] = label0

    if confidence > confidenceLevel:
        img = utils.postprocess_input(img, version=1)
        img = image.array_to_img(img[0])
        # img.save('results/' + str(label) + '-' + str(confidence) + '-' + str(label0) + '-' + str(np.argmax(preds[i])) + '-'+ str(i) + '.jpg')
        outs2.append(np.argmax(output))
        temp = set(outs2)
        flags[label] += 1
        if np.sum(flags!=0)==nb_class:
           totalAttemp = len(outs)
           print("All classes were reached after: " + str(totalAttemp) + " attempts")
           if FinalTotalAttempt==0:
               FinalTotalAttempt=totalAttemp
           break
    if confidence < 0.95:
        failedImage95 += 1
        failedImage99 += 1
    elif confidence < 0.99:
        failedImage99 += 1

if totalAttemp==0:
    totalAttemp = maxAttempt
    FinalTotalAttempt=totalAttemp

#To get the unique labels
temp = set(outs)
print('attempts: ' + str(FinalTotalAttempt))
print(flags)
print("Effectiveness(95): " + str(1 - failedImage95/totalAttemp))
print("Effectiveness(99): " + str(1 - failedImage99/totalAttemp))
