import tensorflow as tf
import numpy as np
from tensorflow import keras

import pandas as pd

df=pd.read_csv('pose_final.csv')

shuffle_df=df.sample(frac=1)


msk = np.random.rand(len(shuffle_df)) < 0.9

train=shuffle_df[msk]

val=shuffle_df[~msk]

train_outputs=train['category']

val_outputs=val['category']


train_inputs=train[['left_shoulder_x','left_shoulder_y','right_shoulder_x','right_shoulder_y','lef_hip_x','lef_hip_y','right_hip_x','right_hip_y',
'left_knee_x','left_knee_y','right_knee_x','right_knee_y','left_ankle_x','left_ankle_y','right_ankle_x','right_ankle_y']]


val_inputs=val[['left_shoulder_x','left_shoulder_y','right_shoulder_x','right_shoulder_y','lef_hip_x','lef_hip_y','right_hip_x','right_hip_y',
'left_knee_x','left_knee_y','right_knee_x','right_knee_y','left_ankle_x','left_ankle_y','right_ankle_x','right_ankle_y']]




model = tf.keras.Sequential([keras.layers.Dense(units=10,activation='relu',input_shape=[16]),
                            keras.layers.Dense(units=10,activation='relu'),
                            keras.layers.Dense(units=8,activation='relu'),
                            keras.layers.Dense(units=4,activation='relu'),
                            keras.layers.Dense(units=4,activation='relu'),
                            keras.layers.Dense(units=1,activation='sigmoid')])





#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


model.fit(train_inputs,train_outputs,epochs=2000)


model.evaluate(val_inputs, val_outputs)