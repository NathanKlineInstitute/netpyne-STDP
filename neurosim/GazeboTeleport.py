#edited: davidd 2022 NKI/Fordham
#original: dlyons@fordham.edu
#NetPyNE robotics project
#Script for moving objects around Gazebo environment

import rospy
import math
import numpy as np
from datetime import datetime
# ROS message definitions
from geometry_msgs.msg import Twist      # ROS Twist message
from nav_msgs.msg import Odometry        # ROS Pose message
from sensor_msgs.msg import LaserScan    # ROS laser msg
from tf.transformations import quaternion_about_axis, euler_from_quaternion
from gazebo_msgs.msg import ModelState #-------------
from gazebo_msgs.srv import GetModelState, SetModelState # Gazebo back door!
_get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
_set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
#--------------
rospy.init_node('control')

class Gazebo:

  def __init__(self,dconf):
    self.dconf = dconf
    self.OBJECTNAME='image0' #name of object in gazebo
    self.referenceFrame = 'world' # ROS world reference frame 
    self.my_get_model_state = _get_model_state(self.OBJECTNAME, self.referenceFrame)#call on object
    self.my_model_state = ModelState()
    self.OriginalPos=dconf['Gazebo']['OriginalPos']# original positions for images in gazebo
    self.TargetPos=[dconf['Gazebo']['TargetPos']["x"],dconf['Gazebo']['TargetPos']["y"]]
    self.done = False

  def loadModelState(self):
    self.my_model_state = ModelState()
    rospy.wait_for_service('/gazebo/get_model_state')
    self.my_model_state.pose = self.my_get_model_state.pose
    self.my_model_state.twist = self.my_get_model_state.twist
    self.my_model_state.reference_frame = self.referenceFrame
    self.my_model_state.model_name = self.OBJECTNAME
    orient = self.my_model_state.pose.orientation
    quat = [axis for axis in [orient.x, orient.y, orient.z, orient.w]]
    (roll,pitch,yaw)=euler_from_quaternion(quat)
    return yaw

  def storeModelState(self):
    rospy.wait_for_service('/gazebo/set_model_state')
    _set_model_state(self.my_model_state)
    return

  def spinModelState(self,angle): # angle in radians
    self.loadModelState() # update the my_model_state variable
    #change the angle
    orient = self.my_model_state.pose.orientation
    quat = [axis for axis in [orient.x, orient.y, orient.z, orient.w]]
    (roll,pitch,yaw)=euler_from_quaternion(quat)
    yaw = angle
    quat = quaternion_about_axis(yaw, (0, 0, 1))
    # assumes no roll or tilt
    self.my_model_state.pose.orientation.x = quat[0]
    self.my_model_state.pose.orientation.y = quat[1]
    self.my_model_state.pose.orientation.z = quat[2]
    self.my_model_state.pose.orientation.w = quat[3]
    self.storeModelState() # tell gazebo to do this
    rospy.sleep(1)
    print("spinModelState: ",self.loadModelState()*(180.0/math.pi))
    return

  def poseModelState(self,x,y,angle): # angle in radians
    self.loadModelState() # update the my_model_state variable
    #change the angle
    pos = self.my_model_state.pose.position
    pos.x, pos.y, pos.z = x, y, 0.0
    orient = self.my_model_state.pose.orientation
    quat = [axis for axis in [orient.x, orient.y, orient.z, orient.w]]
    (roll,pitch,yaw)=euler_from_quaternion(quat)
    yaw = angle
    quat = quaternion_about_axis(yaw, (0, 0, 1))
    # assumes no roll or tilt
    self.my_model_state.pose.orientation.x = quat[0]
    self.my_model_state.pose.orientation.y = quat[1]
    self.my_model_state.pose.orientation.z = quat[2]
    self.my_model_state.pose.orientation.w = quat[3]
    self.storeModelState() # tell gazebo to do this
    rospy.sleep(1)
    return
	
  def resetPositions(self): #returns all training blocks to oringinal position
    for i in np.arange(1,13): 
      self.OBJECTNAME='image'+str(i)
      self.my_get_model_state = _get_model_state(self.OBJECTNAME, self.referenceFrame) #focus on new object
      print('Moving ',self.OBJECTNAME,'to:',self.OriginalPos[self.OBJECTNAME])
      self.poseModelState(self.OriginalPos[self.OBJECTNAME][0],self.OriginalPos[self.OBJECTNAME][1],0) #Reset to initial position
      rospy.sleep(1)
    return
    
  def showRandImage(self):
    imageID = np.random.randint(1,13) #random integar
    self.OBJECTNAME='image'+str(imageID)
    self.my_get_model_state = _get_model_state(self.OBJECTNAME, self.referenceFrame) #focus on new object
    self.poseModelState(self.TargetPos[0],self.TargetPos[1],0) # move in front of robot     
    rospy.sleep(5) #Hold image there for 5 seconds
    return(self.OBJECTNAME, self.OriginalPos[self.OBJECTNAME][2]) # returns image name in ros and string of image class
     
  def moveImageBack(self, name):
    self.OBJECTNAME=name
    self.my_get_model_state = _get_model_state(self.OBJECTNAME, self.referenceFrame) #focus on new object
    self.poseModelStatess(self.OriginalPos[self.OBJECTNAME][0],self.OriginalPos[self.OBJECTNAME][1],0) # move back
    rospy.sleep(1) #Show nothing there for 1 seconds
    return
