import sys
import cv2
import math
import time
import argparse
from openvino.inference_engine import IECore, IENetwork


def post_process(infer):
	age = math.floor(100*infer["age_conv3"][0][0][0][0])
	if infer["prob"][0][0][0][0] > 0.5:
		gender = 'female'
	else:
		gender = 'male'

	return age, gender

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="/home/skellig/Pictures/not_a_person.jpeg", type=str,
	help="path to input image for inference")
ap.add_argument("-p", "--processor", default='CPU', type=str,
	help="processor to use")
args = vars(ap.parse_args())


model = "/home/skellig/Documents/models/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml"
weights = "/home/skellig/Documents/models/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin"

processor = args['processor']
img_path = args['image']
org_image = cv2.imread(img_path)

ie = IECore()

net = ie.read_network(model=model, weights=weights)
input_layer= net.input_info['data'].input_data


#sys.exit(0)

n,c,h,w = input_layer.shape

image = cv2.resize(org_image, (w,h))
image = image.reshape((n,c,h,w))

exec_net = ie.load_network(network=net, device_name=processor)

start_time = time.time()
infer = exec_net.infer({'data': image})
infer_time = time.time() - start_time

age, gender = post_process(infer)

cv2.putText(org_image, f'Approximate Age: {age}', (20, 25),
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
cv2.putText(org_image, f'Gender: {gender}', (20, 55),
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
cv2.putText(org_image, f'Processor: {processor}', (20, 85),
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
cv2.putText(org_image, f'Inference Time: {round(1000*infer_time,3)} ms' , (20, 115),
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

cv2.imshow('fig_1', org_image)
print("Estimated Age: " + str(age))
print("Estimated Gender: " + gender)
print("Inference Time: " + str(infer_time))

cv2.waitKey(0)
cv2.destroyAllWindows()

