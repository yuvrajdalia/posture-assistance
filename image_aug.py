import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(height_shift_range=0.05, 
zoom_range=0.1, horizontal_flip=True)
for i in range(1,30):
	image_path = str('/home/yuvi/projects/minorproject/openpose/bad/bad_'+str(i)+'.png')


	image = np.expand_dims(ndimage.imread(image_path), 0)

	save_here = '/home/yuvi/projects/minorproject/openpose/good_aug'

	datagen.fit(image)

	for x, val in zip(datagen.flow(image,                    #image we chose
        	save_to_dir=save_here,     #this is where we figure out where to save
         	save_prefix=str('bad_aug'+str(i)),        # it will save the images as 'aug_0912' some number for every new augmented image
        	save_format='png'),range(2)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
		pass