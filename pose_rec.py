import cv2
import mediapipe as mp
import math

import numpy as np
import targeting_tools as tt
from pose import Pose, Joint
from matplotlib import pyplot as plt

NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

width = 1920
height = 1080
  
total_joints = {
    0 : "NOSE", 1 : "LEFT_EYE_INNER",2 : "LEFT_EYE",3 : "LEFT_EYE_OUTER",
    4 : "RIGHT_EYE_INNER",5 : "RIGHT_EYE",6 : "RIGHT_EYE_OUTER",
    7 : "LEFT_EAR",8 : "RIGHT_EAR",9 : "MOUTH_LEFT",
    10 : "MOUTH_RIGHT",11 : "LEFT_SHOULDER",12 : "RIGHT_SHOULDER",
    13 : "LEFT_ELBOW",14 : "RIGHT_ELBOW",15 : "LEFT_WRIST",
    16 : "RIGHT_WRIST",17 : "LEFT_PINKY",18 : "RIGHT_PINKY",
    19 : "LEFT_INDEX",20 : "RIGHT_INDEX",21 : "LEFT_THUMB",
    22 : "RIGHT_THUMB",23 : "LEFT_HIP",24 : "RIGHT_HIP",
    25 : "LEFT_KNEE",26 : "RIGHT_KNEE",27 : "LEFT_ANKLE",
    28 : "RIGHT_ANKLE",29 : "LEFT_HEEL",30 : "RIGHT_HEEL",
    31 : "LEFT_FOOT_INDEX",32 : "RIGHT_FOOT_INDEX"
}

key_joints = {
    11 : "LEFT_SHOULDER",
    12 : "RIGHT_SHOULDER",
    13 : "LEFT_ELBOW",
    14 : "RIGHT_ELBOW",
    15 : "LEFT_WRIST",
    16 : "RIGHT_WRIST",
    23 : "LEFT_HIP",
    24 : "RIGHT_HIP",
    25 : "LEFT_KNEE",
    26 : "RIGHT_KNEE",
    27 : "LEFT_ANKLE",
    28 : "RIGHT_ANKLE",
    29 : "LEFT_HEEL",
    30 : "RIGHT_HEEL",
}

key_joints_idx = [11,12,13,14,15,16,23,24,25,26,27,28,29,30]

visibility_threshold = 0.9
circle_radius = 10
circle_color = (128, 0, 255)
circle_left = (255,0,0)
circle_right = (0,255,0)




    
        
# FUNCTIONS

def distance_2(p1:(float,float, float), p2:(float,float, float)):
    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + ((p2[2] - p1[2])**2))
    return distance


def obtain_joints(image, image_name):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode = True) as pose:
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #blank = cv2.imread("blank.jpg")
        #blank_rgb = cv2.cvtColor(blank,cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)
        if result.pose_landmarks is not None:                
            for i in key_joints_idx:
                x1 = int(result.pose_landmarks.landmark[i].x*width)
                y1 = int(result.pose_landmarks.landmark[i].y*height)
                z1 = 0
                visibility = result.pose_landmarks.landmark[i].visibility
                if visibility > visibility_threshold:
                    cv2.circle(image_rgb, (x1,y1), circle_radius, circle_color , -1)   

            cv2.imwrite("marked_" + image_name , image_rgb)
            
            return result



def snap_picture(idx1,idx2, name_1, name_2):

    cam1 = cv2.VideoCapture(idx1)
    cam2 = cv2.VideoCapture(idx2)

    result1, image1 = cam1.read()
    result2, image2 = cam2.read()

    img1 = None
    img2 = None

    if result1 and result2:
        cv2.imwrite(name_1, image1)
        cv2.imwrite(name_2, image2)
        img1 = obtain_joints(image1, name_1)
        img2 = obtain_joints(image2, name_2)

    cam1.release() 
    cam2.release() 

    return image1, image2




def find_depth(image1, image2, result1, result2, name_1, name_2, joint_index, joint_name):

    pixel_width = 1920
    pixel_height = 1080
    angle_width = 78
    angle_height = 64 
    frame_rate = 20
    camera_separation = 3 + 15/16
    width = 1920
    height = 1080

    frame1 = image1
    frame2 = image2
    angler = tt.Frame_Angles(pixel_width,pixel_height,angle_width,angle_height)
    angler.build_frame()

    x1m = int(result1.pose_landmarks.landmark[joint_index].x*width)
    y1m = int(result1.pose_landmarks.landmark[joint_index].y*height)
    x2m = int(result2.pose_landmarks.landmark[joint_index].x*width)
    y2m = int(result2.pose_landmarks.landmark[joint_index].y*height)
    # get angles from camera centers
    xlangle,ylangle = angler.angles_from_center(x1m,y1m,top_left=True,degrees=True)
    xrangle,yrangle = angler.angles_from_center(x2m,y2m,top_left=True,degrees=True)
    # triangulate
    X,Y,Z,D = angler.location(camera_separation,(xlangle,ylangle),(xrangle,yrangle),center=True,degrees=True)
    # display camera centers
    angler.frame_add_crosshairs(frame1)
    angler.frame_add_crosshairs(frame2)

    cv2.circle(frame1, (x1m,y1m), 10, circle_color , -1)   
    text = 'X: {:3.1f}\nY: {:3.1f}\nZ: {:3.1f}\nD: {:3.1f}'.format(X,Y,Z,D)
    lineloc = 0
    lineheight = 30
    for t in text.split('\n'):
        lineloc += lineheight
        cv2.putText(frame1,t,(10,lineloc),cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0),1,cv2.LINE_AA,False) 

    cv2.imwrite("depth_" + joint_name + "_" + name_1 , frame1)
    #cv2.imwrite("depth_" + joint_name + "_" + name_2 , frame2)
    return Z

# FUNCTIONS

exp = True
name1 = name2 = None
im1 = im2 = None
z_scaling = 0.9

if not exp:
    name1 = "img1.png"
    name2 = "img2.png"
    im1, im2 = snap_picture(0,1, name1, name2)
else:
    name1 = "img1_pose1.png"
    name2 = "img2_pose1.png"
    im1 = cv2.imread(name1)
    im2 = cv2.imread(name2)

pose = Pose(key_joints, visibility_threshold)

result1 = obtain_joints(im1, name1)
result2 = obtain_joints(im2, name2)

if result1 and result2:
    for idx, name in key_joints.items():
        if result1.pose_landmarks.landmark[idx].visibility > visibility_threshold and result2.pose_landmarks.landmark[idx].visibility > visibility_threshold:
            z = find_depth(im1,im2,result1, result2, name1, name2, idx, name)
            x = int(result1.pose_landmarks.landmark[idx].x*width)
            y = int(result1.pose_landmarks.landmark[idx].y*height)
            visibility = result1.pose_landmarks.landmark[idx].visibility
            print(name, "X:",x,"Y:", y,"Z:",  z)

            joint = Joint(x,y,z*z_scaling, visibility, idx)
            pose.add_joint(joint)

pose.generate_video(circle_radius, circle_left, circle_right, circle_color, width, height)




