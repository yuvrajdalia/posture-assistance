from matplotlib import pyplot as plt
import pandas as pd
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

# Note that we can reset the classes of the detector to only include
# human, so that the NMS process is faster.

detector.reset_class(["person"], reuse_weights=['person'])

good_images=['good_3.png','good_13.png','good_25.png','bad_5.png','bad_7.png','bad_35.png']
coords=[]
df=pd.DataFrame()
for good_image in good_images:
	x, img = data.transforms.presets.ssd.load_test(good_image, short=512)
	print('Shape of pre-processed image:', x.shape)

	class_IDs, scores, bounding_boxs = detector(x)

	pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

	predicted_heatmap = pose_net(pose_input)

	pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
	arr1=pred_coords[0][5:7]
	arr2=pred_coords[0][11:17]
	print(arr1)
	print(arr2)
	arr3=(arr1.squeeze().asnumpy().tolist()+arr2.squeeze().asnumpy().tolist())
	coords.append(arr3)
	print(arr3[0][0],arr3[0][1],arr3[1][0],arr3[1][1],)
	df=df.append({'left_shoulder_x':arr3[0][0],'left_shoulder_y':arr3[0][1],'right_shoulder_x':arr3[1][0],'right_shoulder_y':arr3[1][1],'lef_hip_x':arr3[2][0],'lef_hip_y':arr3[2][1],'right_hip_x':arr3[3][0],'right_hip_y':arr3[3][1],'left_knee_x':arr3[4][0],'left_knee_y':arr3[4][1],'right_knee_x':arr3[5][0],'right_knee_y':arr3[5][1],'left_ankle_x':arr3[6][0],'left_ankle_y':arr3[6][1],'right_ankle_x':arr3[7][0],'right_ankle_y':arr3[7][1]
},ignore_index=True)
	ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
	plt.show()
print(coords)
print(df)
df.to_csv('real_test.csv')