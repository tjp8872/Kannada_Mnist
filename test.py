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

train_datas = pd.read_csv("train.csv")
val_datas = pd.read_csv("Dig-MNIST.csv")

print(train_datas.shape)
print(val_datas.shape)
#train_datas = train_datas / 255.0
#val_datas = val_datas / 255.0

#concat train and val
datas = pd.concat([train_datas,val_datas],axis=0)
print(datas.shape)
datas_X = np.array(datas.drop("label",axis=1),dtype=np.float32)
datas_Y = np.array(datas[["label"]],dtype=np.int32)
train_X,val_X,train_Y,val_Y = train_test_split(datas_X,datas_Y,test_size=0.1,shuffle=True)

train_X = np.reshape(train_X,(-1,28,28,1))
val_X = np.reshape(val_X,(-1,28,28,1))

print(train_X.shape,train_X.dtype)
print(train_Y.shape,train_Y.dtype)
print(val_X.shape,val_X.dtype)
print(val_Y.shape,val_Y.dtype)



#28x28
input_layer = Input(shape=(28,28,1))
net = Conv2D(64, (3,3), padding='same')(input_layer)
net = ELU(alpha=0.1)(net)
net = BatchNormalization()(net)


net = Conv2D(64,  (3,3), padding='same')(net)
net = ELU(alpha=0.1)(net)
net = BatchNormalization()(net)

net = MaxPooling2D(2, 2)(net)
net = Dropout(0.2)(net)

net = Conv2D(128,  (3,3), padding='same')(net)
net = ELU(alpha=0.1)(net)
net = BatchNormalization()(net)

net = Conv2D(128,  (3,3), padding='same')(net)
net = ELU(alpha=0.1)(net)
net = BatchNormalization()(net)



net = MaxPooling2D(2, 2)(net)


net = Flatten()(net)
net = Dense(256, name="Dense1")(net)
net = ELU(alpha=0.1)(net)
net = BatchNormalization()(net)

net = Flatten()(net)
net = Dense(256, name="Dense2")(net)
net = ELU(alpha=0.1)(net)
net = BatchNormalization()(net)

output = Dense(10, activation='softmax')(net)


model = Model(inputs=input_layer,outputs=output)
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss="sparse_categorical_crossentropy",metrics=["accuracy"])

datagen = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 0.1,
    zoom_range = 0.25,
    horizontal_flip = False,
    rescale=1/255.0)

epochs = 18
batch_size = 102


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=10, 
                                            factor=0.2, 
                                            min_lr=0.000001)

earlyStopping = EarlyStopping(monitor='val_acc', patience=16)

history = model.fit_generator(datagen.flow(train_X,train_Y, batch_size=batch_size),
                                    epochs = epochs, 
                                    steps_per_epoch = 30,
                                    validation_data = (val_X/255.0,val_Y),
                                    callbacks=[earlyStopping,learning_rate_reduction],
                                    )


