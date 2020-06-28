from keras.models import model_from_json
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd



df=pd.read_csv('real_test.csv')



# msk = np.random.rand(len(shuffle_df)) < 0.9

# train=shuffle_df[msk]

# val=shuffle_df[~msk]


# val_outputs=val['category']


test_inputs=df[['left_shoulder_x','left_shoulder_y','right_shoulder_x','right_shoulder_y','lef_hip_x','lef_hip_y','right_hip_x','right_hip_y',
'left_knee_x','left_knee_y','right_knee_x','right_knee_y','left_ankle_x','left_ankle_y','right_ankle_x','right_ankle_y']]


# val_inputs=val[['left_shoulder_x','left_shoulder_y','right_shoulder_x','right_shoulder_y','lef_hip_x','lef_hip_y','right_hip_x','right_hip_y',
# 'left_knee_x','left_knee_y','right_knee_x','right_knee_y','left_ankle_x','left_ankle_y','right_ankle_x','right_ankle_y']]


print(test_inputs.iloc[0].to_numpy())
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = loaded_model.evaluate(test_inputs, test_outputs, verbose=0)
# print("Accuracy on test set:  ")
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
res=(loaded_model.predict(test_inputs))
j=1
for i in res:
	print(" For image : ",j)
	if(i>0.1):
		print("Amazing, Your form is correct")
	else:
		print("You need to work on your form")
	j=j+1