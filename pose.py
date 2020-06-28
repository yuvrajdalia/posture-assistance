from matplotlib import pyplot as plt
import numpy as np
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

# Note that we can reset the classes of the detector to only include
# human, so that the NMS process is faster.

detector.reset_class(["person"], reuse_weights=['person'])


im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/pose/soccer.png?raw=true',
                          path='soccer.png')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

class_IDs, scores, bounding_boxs = detector(x)

pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

predicted_heatmap = pose_net(pose_input)

pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
print(pred_coords)
arr1=pred_coords[0][5:7]
arr2=pred_coords[0][11:17]
arr3=(arr1.squeeze().asnumpy().tolist()+arr2.squeeze().asnumpy().tolist())
print(arr3)
ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2)
plt.show()