import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense,ELU, Flatten, Dropout, BatchNormalization,Add,GlobalAveragePooling2D,Softmax, Concatenate,add
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

from sklearn.model_selection import train_test_split
import random
from PIL import Image

train_datas = pd.read_csv("train.csv")
val_datas = pd.read_csv("Dig-MNIST.csv")

print(train_datas.shape)
print(val_datas.shape)
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#train_datas = train_datas / 255.0
#val_datas = val_datas / 255.0

#concat train and val
datas = pd.concat([train_datas,val_datas],axis=0)
print(datas.shape)
datas_X = np.array(datas.drop("label",axis=1),dtype=np.float32)
datas_Y = np.array(datas[["label"]],dtype=np.int32)
train_X,val_X,train_Y,val_Y = train_test_split(datas_X,datas_Y,test_size=0.2,shuffle=True)

train_X = np.reshape(train_X,(-1,28,28,1))
val_X = np.reshape(val_X,(-1,28,28,1))

print(train_X.shape,train_X.dtype)
print(train_Y.shape,train_Y.dtype)
print(val_X.shape,val_X.dtype)
print(val_Y.shape,val_Y.dtype)




row,col,ch= (28,28,1)
mean = 0
var = 7 
sigma = var**0.65
gauss = np.random.normal(mean,sigma,(train_X.shape[0],row,col,ch))
gauss = gauss.reshape(train_X.shape[0],row,col,ch)
noisy = train_X + gauss
noisy[noisy>255] = 255
noisy[noisy<0] = 0


mean = 0
var = 10
sigma = var**0.4
gauss = np.random.normal(mean,sigma,(train_X.shape[0],row,col,ch))
gauss = gauss.reshape(train_X.shape[0],row,col,ch)
noisy2 = train_X + gauss
noisy2[noisy2>255] = 255
noisy2[noisy2<0] = 0

tmp_y = train_Y

train_X = np.concatenate((train_X,noisy))
train_Y = np.concatenate((train_Y,tmp_y))

train_X = np.concatenate((train_X,noisy2))
train_Y = np.concatenate((train_Y,tmp_y))






print(train_X.shape,train_X.dtype)
print(train_Y.shape,train_Y.dtype)
print(val_X.shape,val_X.dtype)
print(val_Y.shape,val_Y.dtype)


#28x28
input_layer = Input(shape=(28,28,1))
net = Conv2D(64, (3,3), padding='same')(input_layer)
net = BatchNormalization()(net)
net = ELU(alpha=0.1)(net)



net = Conv2D(64,  (3,3), padding='same')(net)
net = BatchNormalization()(net)
net = ELU(alpha=0.1)(net)


net = MaxPooling2D(2, 2)(net)
net = Dropout(0.2)(net)

net = Conv2D(128,  (3,3), padding='same')(net)
net = BatchNormalization()(net)
net = ELU(alpha=0.1)(net)


net = Conv2D(128,  (3,3), padding='same')(net)
net = BatchNormalization()(net)
net = ELU(alpha=0.1)(net)




net = MaxPooling2D(2, 2)(net)
net = Dropout(0.2)(net)

net = Conv2D(196,  (3,3), padding='same')(net)
net = BatchNormalization()(net)
net = ELU(alpha=0.1)(net)


net = Conv2D(196,  (3,3), padding='same')(net)
net = BatchNormalization()(net)
net = ELU(alpha=0.1)(net)

net = MaxPooling2D(2, 2)(net)
net = Dropout(0.2)(net)


net = Flatten()(net)
net = Dense(96, name="Dense1")(net)
net = BatchNormalization()(net)
net = ELU(alpha=0.1)(net)


net = Dense(96, name="Dense2")(net)
net = BatchNormalization()(net)
net = ELU(alpha=0.1)(net)s

output = Dense(10, activation='softmax')(net)


model = Model(inputs=input_layer,outputs=output)
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss="sparse_categorical_crossentropy",metrics=["accuracy"])

datagen = ImageDataGenerator(
    rotation_range = 7,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    shear_range = 0.15,
    zoom_range = 0.15,
    horizontal_flip = False,
    rescale=1/255.0)

epochs = 120
batch_size = 1024


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=5, 
                                            factor=0.5, 
                                            min_lr=0.000001,
                                            )

#earlyStopping = EarlyStopping(monitor='val_acc', patience=10)

history = model.fit_generator(datagen.flow(train_X,train_Y, batch_size=batch_size),
                                    epochs = epochs, 
                                    #steps_per_epoch = math.ceil(train_X.shape[0]/batch_size)*2,
                                    validation_data = (val_X/255.0,val_Y),
                                    callbacks=[learning_rate_reduction],
                                    )


