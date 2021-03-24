import os
import sys

import cv2
import rosbag
from cv_bridge import CvBridge, CvBridgeError

filepath = sys.argv[1]
bag = rosbag.Bag(filepath)

# Create directory with name filename (without extension)
path, filename = os.path.split(filepath)
results_dir = filepath[:-4] + "_data"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

images_dir = results_dir + "/images"
timestamp_dir = results_dir + "/Timestamps"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
if not os.path.exists(timestamp_dir):
    os.makedirs(timestamp_dir)

print("Writing image data to dir")

outfile = open(timestamp_dir + '/' + filename[:-4] + '.txt', 'w')

bridge = CvBridge()
timestamp_list = []
clahe = cv2.createCLAHE(100, (8, 8))
count = 0
for topic, msg, t in bag.read_messages(topics=['/thermal/image_raw']):
    # message of type sensor_msgs/Image
    # uncooled: mono16, /ubol/image_raw
    # cooled: mono16, /t2sls/image_raw
    # rgb: mono8, /rgb/image_raw
    # print(msg.height, msg.width, msg.encoding, msg.step)

    try:
        input_image = bridge.imgmsg_to_cv2(msg)
        clahe_image = clahe.apply(input_image)
        output_image = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # Saving the image using timestamp as file name
        timestamp = int(msg.header.stamp.secs * 1e9) + int(msg.header.stamp.nsecs)
#        print(timestamp)
        img_filename = str(count) + '.tiff'
        cv2.imwrite(images_dir + '/' + img_filename, output_image)
        timestamp_list.append(timestamp)
	count += 1

    except Exception as e:
        print(e)

timestamp_list = list(set(timestamp_list))
timestamp_list.sort()
outfile.write('\n'.join(str(time) for time in timestamp_list))
outfile.close()
