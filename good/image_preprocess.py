from matplotlib import pyplot as plt
import os
import pandas as pd
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

# Note that we can reset the classes of the detector to only include
# human, so that the NMS process is faster.

detector.reset_class(["person"], reuse_weights=['person'])
df = pd.DataFrame()

good_images=os.listdir('/home/yuvi/projects/minorproject/openpose/good')
coords=[]
for good_image in good_images:
	if (good_image != 'image_preprocess.py'):
		x, img = data.transforms.presets.ssd.load_test(good_image, short=512)
		print('Shape of pre-processed image:', x.shape)
		print(good_image)
		class_IDs, scores, bounding_boxs = detector(x)

		pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

		predicted_heatmap = pose_net(pose_input)

		pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
		arr1=pred_coords[0][5:7]
		arr4=[]
		for i in arr1:
			arr4.append(i[0].asnumpy()[0])
			arr4.append(i[1].asnumpy()[0])
		print(arr4)
		print(arr1)
		arr2=pred_coords[0][11:17]
		for i in arr2:
			arr4.append(i[0].asnumpy()[0])
			arr4.append(i[1].asnumpy()[0])
		print(arr4)
		df=df.append({'left_shoulder_x':arr4[0],'left_shoulder_y':arr4[1],'right_shoulder_x':arr4[2],'right_shoulder_y':arr4[3],
			'lef_hip_x':arr4[4],'lef_hip_y':arr4[5],'right_hip_x':arr4[6],'right_hip_y':arr4[7],'left_knee_x':arr4[8],'left_knee_y':arr4[9],'right_knee_x':arr4[10],
			'right_knee_y':arr4[11],'left_ankle_x':arr4[12],'left_ankle_y':arr4[13],'right_ankle_x':arr4[14],'right_ankle_y':arr4[15],'category':1},ignore_index=True)
		#df.append(new_row, ignore_index=True)
		print(arr2)
		arr5=[1,1,1,1,1,1,1,1]
		arr3=(arr1.squeeze().asnumpy().tolist()+arr2.squeeze().asnumpy().tolist())
		coords.append(arr3)
		#print(df.head)
		'''ax = utils.viz.plot_keypoints(img,pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
		plt.show()'''
print(df.head)
df.to_csv(r'./pose.csv', index = False)