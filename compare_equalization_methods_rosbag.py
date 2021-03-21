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
import matplotlib.pyplot as plt

filepath = sys.argv[1]
bag = rosbag.Bag(filepath)

bridge = CvBridge()
count = 0
clahe_vals = [1, 40, 100, 200, 300, 500]
for topic, msg, t in bag.read_messages(topics=['/ubol/image_raw']):

    try:
        if count % 100 == 0:
            input_image = bridge.imgmsg_to_cv2(msg)
            img_fig, img_axes = plt.subplots(1, len(clahe_vals))
            for i in range(len(clahe_vals)):
                clahe = cv2.createCLAHE(clahe_vals[i], (8, 8))
                clahe_image = clahe.apply(input_image)
                output_image = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                img_axes[i].imshow(output_image, cmap='gray', vmin=0, vmax=255)
                img_axes[i].set_title('CLAHE: Clip Limit ' + str(clahe_vals[i]))
            plt.show()

        count += 1
    except Exception as e:
        print(e)
