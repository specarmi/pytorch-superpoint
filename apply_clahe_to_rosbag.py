import os
import sys
import time
import rospy
import cv2
import rosbag
from cv_bridge import CvBridge, CvBridgeError
import roslib
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image

filepath = sys.argv[1]
bag = rosbag.Bag(filepath)

# Create directory with name filename (without extension)
path, filename = os.path.split(filepath)
results_dir = filepath[:-4] + "_data"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

bridge = CvBridge()
count = 0
clahe = cv2.createCLAHE(100, (8, 8))
new_bag = rosbag.Bag('new_bag', 'w')
for topic, msg, t in bag.read_messages(topics=['/thermal/image_raw']):

    try:
        input_image = bridge.imgmsg_to_cv2(msg)
        clahe_image = clahe.apply(input_image)
        output_image = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        img_msg = Image()
        img_msg = bridge.cv2_to_imgmsg(output_image)
        img_msg.header.seq = count
        img_msg.header.stamp = msg.header.stamp
        img_msg.header.frame_id = "camera"

        new_bag.write('camera/image_raw', img_msg, msg.header.stamp)

        count += 1
    except Exception as e:
        print(e)
new_bag.close()