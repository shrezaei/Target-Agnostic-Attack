from keras_vggface.vggface import VGGFace
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

RetrainAllLayers = False
newLayers = 1
nb_class = 20
target_class = 5

# np.random.seed(int(sys.argv[1]))
#newLayers = int(sys.argv[2])

x_train_all = np.load('trainData.npy')
y_train_all = np.load('trainLabel.npy')
x_test_all = np.load('testData.npy')
y_test_all = np.load('testLabel.npy')

#Radnomly select several target classes to re-train the model
train_set_ids = []
while len(train_set_ids) < target_class:
    new_id = np.random.randint(19) + 1
    if new_id in train_set_ids:
        continue
    train_set_ids.append(new_id)

train_set = y_train_all==-1     #create a list with all False elements
test_set = y_test_all==-1
for id in train_set_ids:
    train_set = train_set | (y_train_all==id)
    test_set = test_set | (y_test_all==id)

x_train = x_train_all[train_set]
y_train = y_train_all[train_set]
x_test = x_test_all[test_set]
y_test = y_test_all[test_set]

#change the range of target class labels to (1, n)
for i in range(0,y_train.shape[0]):
    y_train[i] = train_set_ids.index(y_train[i]) + 1
for i in range(0, y_test.shape[0]):
    y_test[i] = train_set_ids.index(y_test[i]) + 1

temp = np.zeros((y_train.shape[0], int(np.max(y_train))))
temp[np.arange(y_train.shape[0]), y_train.astype(int) - 1] = 1
y_train = temp

temp = np.zeros((y_test.shape[0], int(np.max(y_test))))
temp[np.arange(y_test.shape[0]), y_test.astype(int) - 1] = 1
y_test = temp

if newLayers==1:
    model = VGGFace(model='vgg16', pooling='max')
    last_layer = model.get_layer('fc7/relu').output
    x = Dense(target_class, name='fc8-2')(last_layer)
    out = Activation('softmax', name='fc8-2/softmax')(x)
    custom_model = Model(model.input, out)
elif newLayers==2:
    model = VGGFace(model='vgg16', pooling='max')
    last_layer = model.get_layer('fc7/relu').output
    x = Dense(4096, name='fc8-2')(last_layer)
    x = Activation('relu', name='fc8-2/relu')(x)
    x = Dense(target_class, name='fc8-2-1')(x)
    out = Activation('softmax', name='fc8-2/softmax')(x)
    custom_model = Model(model.input, out)
elif newLayers==3:
    model = VGGFace(model='vgg16', pooling='max')
    last_layer = model.get_layer('fc7/relu').output
    x = Dense(4096, name='fc8-2')(last_layer)
    x = Activation('relu', name='fc8-2/relu')(x)
    x = Dense(4096, name='fc8-2-1')(x)
    x = Activation('relu', name='fc8-2-1/relu')(x)
    x = Dense(target_class, name='fc8-2-2')(x)
    out = Activation('softmax', name='fc8-2/softmax')(x)
    custom_model = Model(model.input, out)

# custom_model.summary()

for layer in custom_model.layers:
    layer.trainable = True

if not RetrainAllLayers:
    for layer in custom_model.layers[:-2 - 2*(newLayers)]:
        layer.trainable = False

custom_model.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
#custom_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

custom_model.fit(x=x_train, y=y_train, epochs=20, batch_size=32)

result = custom_model.evaluate(x_test, y_test)
print(result)

custom_model.save('vggface-retrained.h5')

