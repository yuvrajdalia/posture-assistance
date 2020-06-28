import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
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


history=model.fit(train_inputs,train_outputs,epochs=2000,validation_data=(val_inputs, val_outputs))
loss=(history.history['loss'])
acc=(history.history['acc'])
val_loss=history.history['val_loss']
val_acc=history.history['val_acc']
# epochs = range(1,2001)
# plt.plot(epochs, acc, 'g', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# print(acc/2000)
print("Highest accuracy achived in training set: ",np.amax(acc))
print("Highest accuracy achived in validation set: ",np.amax(val_acc))
#history_test=model.evaluate(val_inputs, val_outputs)

# print("Accuracy achieved in test set : ",history_test[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")