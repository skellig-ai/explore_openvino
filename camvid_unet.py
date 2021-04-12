import sys
import cv2
import math
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore, IENetwork

def post_process(infer):
	n, c, h, w = infer['206'].shape
	infer_res = np.round(infer['206'].reshape(h,w,c,n))
	for idx in range(0,c):
		infer_res[:,:,idx,:] = idx*infer_res[:,:,idx,:]
	return infer_res

path = Path.cwd()
parent_path = path.parent
image_path = path/'images'
model_path = parent_path/'openvino_models'/'intel'

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default=image_path/"camvid_example.png", type=str,
		help='path to image file')
ap.add_argument('-d', '--device', default='CPU', type=str,
		help='processor to use, default is CPU')
args = vars(ap.parse_args())

device = args['device']
img_path = args['image']

org_image = cv2.imread(str(img_path))

model = model_path/'unet-camvid-onnx-0001/FP16/unet-camvid-onnx-0001.xml'
weights = model_path/'unet-camvid-onnx-0001/FP16/unet-camvid-onnx-0001.bin'

ie = IECore()

net = ie.read_network(model=model, weights=weights)
input_layer= net.input_info['input.1'].input_data

#print(input_layer.shape)
#sys.exit(0)

n,c,h,w = input_layer.shape

org_image = cv2.resize(org_image, (w,h))
image = org_image.reshape((n,c,h,w))


exec_net = ie.load_network(network=net, device_name = device)

infer = exec_net.infer({'input.1': image})
post_processed_image = np.argmax(infer['206'],axis=1).reshape((h,w,n))
print(infer['206'].shape)
print(post_processed_image.shape)
#sys.exit(0)
#infer_res = post_process(infer)
#print(np.unravel_index(np.max(infer_res,axis=2),infer_res.shape))
#sys.exit(0)

#plt.figure()
plt.imshow(org_image, cmap='gray') # I would add interpolation='none'
plt.imshow(post_processed_image, cmap='jet', alpha=0.5) # interpolation='none'
plt.show()
#cv2.imshow('fig_1', post_processed_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print()

