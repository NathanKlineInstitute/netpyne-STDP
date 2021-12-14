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

class ROSImage:

	def __init__(self):
		self.currentImage = Image() # global varibale for image
		self.Bridge = CvBridge() # make a ROS to CV bridge object
		self.imageTopic = '/camera/rgb/image_raw' # ROS topic name
		self.done = False

	# Callback for image topic
	def callbackImage(self, img): #Called automatically for each new image
		self.currentImage = self.Bridge.imgmsg_to_cv2(img, 'bgr8')
		return
		
	# Display whatever images are present
	def displayNode(self):
		#make a CV2 window to display image
		cv2.namedWindow('Turtlebot Camera', cv2.WINDOW_NORMAL)	
		cv2.imshow('Turtlebot Camera', self.currentImage)
		cv2.waitKey(1) # HAVE to do this for image to show in window
		return
		
	# Get height and width
	def getImageSize(self,h,w):
		array = self.currentImage.shape
		h = array[0]
		w = array[1]
		return (h,w)

	# Reduce image to 5x5 array
	def processImage(self, pImage, h,w):
		array = np.zeros((h,w))
		h_inc = h/5 #216
		w_inc = w/5 #384
		
		bwImage = np.average(self.currentImage,axis=2) # convert rgb to avg for black & white
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
	def getImage(self):
		print("Initialize program")
		#declare variables 
		height, width = 0,0
		pImage = np.zeros((5,5)) #Processed image is 5x5 array
		print("Declared variables")
		# Collect image from gazebo
		rospy.init_node('displayNode',anonymous=True)
		print("Stop 1")
		# launch the subscriber callback to store the image
		imageSub = rospy.Subscriber(self.imageTopic,Image,self.callbackImage)
		print("Stop 2")
		rospy.sleep(0.5) # wait for callback to catch up
		print("Stop 3")
		rate = rospy.Rate(10)
		print("Stop 4")
		print("Set up ready")
		#ProcessImage
		if not rospy.is_shutdown():
			displayNode()
			height, width = getImageSize(height, width)
			pImage = processImage(pImage, height, width) # 5x5 image of average color
			print(pImage)
			imageList = pImage.reshape(25)
			rate.sleep()
		else:
			self.done = True
		return(imageList)