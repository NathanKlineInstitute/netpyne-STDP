# Final Project
# Input to network: 5x5 visual grid
# Takes camera image from gazebo
# davidd 2021 Fordham University

# Import Libraries
import math
import numpy as np
import rospy
import sys
import cv2 # computer vision
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#ROS topic names

imageTopic = '/camera/rgb/image_raw'

#
# Global variables
#

gCurrentImage = Image() # global varibale for image
gBridge = CvBridge() # make a ROS to CV bridge object

#
#Functions
#

# Callback for image topic
def callbackImage(img): #Called automatically for each new image
	global gCurrentImage, gBridge
	gCurrentImage = gBridge.imgmsg_to_cv2(img, 'bgr8')
	return
	
# Display whatever images are present
def displayNode():
	global gCurrentImage
	#make a CV2 window to display image
	cv2.namedWindow('Turtlebot Camera', cv2.WINDOW_AUTOSIZE)	
	cv2.imshow('Turtlebot Camera', gCurrentImage)
	cv2.waitKey(1) # HAVE to do this for image to show in window
	return
	
	
# Get height and width
def getImageSize(h,w):
	global gCurrentImage
	array = gCurrentImage.shape
	h = array[0]
	w = array[1]
	return (h,w)

# Reduce image to 5x5 array
def processImage(pImage, h,w):
	global gCurrentImage
	array = np.zeros((h,w))
	h_inc = h/5 #216
	w_inc = w/5 #384
	
	bwImage = np.average(gCurrentImage,axis=2) # convert rgb to avg for black & white
	#Average over a slice in 5x5 grid 
	for i in range(5): # height 
		h_start = int(i*h_inc)
		h_end = int((i+1)*h_inc)
		for j in range(5): # width
			w_start = int(j*w_inc)
			w_end = int((j+1)*w_inc)
			pImage[i][j]=np.average(bwImage[h_start:h_end, w_start:w_end])/255 #avg and normalize
	return(pImage)
	

# Procedure to display image as 5x5 grid
def main():
	print("Initialize program")
	#declare variables 
	global gCurrentImage
	height, width = 0,0
	pImage = np.zeros((5,5)) #Processed image is 5x5 array
	print("Declared variables")
	# Collect image from gazebo
	rospy.init_node('displayNode',anonymous=True)
	print("Stop 1")
	# launch the subscriber callback to store the image
	imageSub = rospy.Subscriber(imageTopic,Image,callbackImage)
	print("Stop 2")
	rospy.sleep(0.5) # wait for callback to catch up
	print("Stop 3")
	rate = rospy.Rate(10)
	print("Stop 4")
	print("Set up ready")
	#ProcessImage
	while not rospy.is_shutdown():
		displayNode()
		height, width = getImageSize(height, width)
		pImage = processImage(pImage, height, width) # 5x5 image of average color
		print(pImage)
		imageList = pImage.reshape(25)
		rate.sleep()
	return()

if __name__ == '__main__':
	main()
