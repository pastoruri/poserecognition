import cv2
import mediapipe as mp
import math
import imageio
import os
import numpy as np
import targeting_tools as tt
import numpy as np
import random
import pickle
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from pose import Pose, Joint, Point
from matplotlib import pyplot as plt
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.video import Video
from kivy.uix.camera import Camera
from kivy.uix.videoplayer import VideoPlayer
from pose_rec import *

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
exp = True
name1 = name2 = None
im1 = im2 = im11 = im22 =None
z_scaling = 30

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


possible_edges = [
    ("LEFT_SHOULDER","RIGHT_SHOULDER"),
    ("LEFT_SHOULDER","LEFT_HIP"),
    ("RIGHT_SHOULDER","RIGHT_HIP"),
    ("LEFT_HIP","RIGHT_HIP"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW","LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW","RIGHT_WRIST"),
    ("LEFT_HIP","LEFT_KNEE"),
    ("LEFT_KNEE","LEFT_ANKLE"),
    ("RIGHT_HIP","RIGHT_KNEE"),
    ("RIGHT_KNEE","RIGHT_ANKLE")
]

possible_edges_idx = [
    (11,12),
    (11,23),
    (12,24),
    (23,24),
    (11, 13),
    (13,15),
    (12, 14),
    (14,16),
    (23,25),
    (25,27),
    (24,26),
    (26,28)
]

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

            #cv2.imwrite("marked_" + image_name , image_rgb)
            
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



def project_poses(poses, circle_radius, circle_left, circle_right, width, height, possible_edges, video_name, center = True):


    blank1 = cv2.imread("blank.jpg")
    img_array = []

    for i in range(0,180):
        blank_rgb1 = cv2.cvtColor(blank1,cv2.COLOR_BGR2RGB)
        for tupla in poses:
            
            pose, circle_color = tupla
            cx, cy = pose.find_center()
            pose.key_point = Point(cx,cy,0)
            alt_joints = dict(pose.joints)
            rad_angle =  math.radians(i)
            
            if center:
                for name, joint in alt_joints.items():
                    if joint is not None:
                        new_x= pose.key_point.x + (joint.x - pose.key_point.x) * math.cos(rad_angle) - (joint.z - pose.key_point.z) * math.sin(rad_angle)
                        new_z = pose.key_point.z + (joint.x- pose.key_point.x) * math.sin(rad_angle) + (joint.z - pose.key_point.z) * math.cos(rad_angle)
                        new_joint = Joint(new_x, joint.y, new_z, joint.visibility, joint.id)
                        alt_joints[name] = new_joint
            
            else:
                for name, joint in alt_joints.items():
                    if joint is not None:
                        new_x =  joint.x  * math.cos(rad_angle) - joint.z * math.sin(rad_angle)
                        new_z = joint.x * math.sin(rad_angle) + joint.z * math.cos(rad_angle)
                        new_joint = Joint(new_x, joint.y, new_z, joint.visibility, joint.id)
                        alt_joints[name] = new_joint

            pose.add_circles_independent(blank_rgb1, alt_joints, circle_radius, circle_left, circle_right, circle_color)
            

            for edge in possible_edges:
                if pose.joints[edge[0]] is not None and pose.joints[edge[1]] is not None:
                    if pose.joints[edge[0]].visibility > pose.visibility_threshold and pose.joints[edge[1]].visibility > pose.visibility_threshold:
                        cv2.line(blank_rgb1, (int(alt_joints[edge[0]].x) , int(alt_joints[edge[0]].y)), (int(alt_joints[edge[1]].x ), int(alt_joints[edge[1]].y)), circle_color, 10)
            
        img_array.append(blank_rgb1)

            
    codec = cv2.VideoWriter_fourcc(*'mp4v')     
    out = cv2.VideoWriter(video_name,codec, 36, (width,height))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



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


def compare_poses(pose1 : Pose, pose2 : Pose, tolerance):
    same_pose = True
    for idx, name in key_joints.items():
        if pose1.joints[name] is not None and pose2.joints[name] is not None:
            if pose1.joints[name].visibility > pose1.visibility_threshold and pose2.joints[name].visibility > pose2.visibility_threshold:
                if pose1.joint_distance(pose1.joints[name], pose2.joints[name]) > tolerance:
                    same_pose = False
    return same_pose


def map_pose(im1, im2, name1, name2):

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
    return pose


def KD_distance(pose1 : Pose, pose2 : Pose):

    puntos1 = []
    puntos2 = []
    for idx, name in key_joints.items():
        if pose1.joints[name] is not None and pose2.joints[name] is not None:
            if pose1.joints[name].visibility > pose1.visibility_threshold and pose2.joints[name].visibility > pose2.visibility_threshold:
                
                
                x1 = pose1.joints[name].x
                y1 = pose1.joints[name].y
                z1 = pose1.joints[name].z
                puntos1.append([x1,y1,z1])

                x2 = pose2.joints[name].x
                y2 = pose2.joints[name].y
                z2 = pose2.joints[name].z
                puntos2.append([x2,y2,z2])

    nube_puntos1 = np.array(puntos1)
    nube_puntos2 = np.array(puntos2)
    tree1 = cKDTree(nube_puntos1)
    tree2 = cKDTree(nube_puntos2)
    distancias = []
    for punto1 in puntos1:
        distancia, _ = tree2.query(punto1)
        distancias.append(distancia)

    distancia_promedio = np.mean(distancias)

    return distancia_promedio


def wasserstein_pose_distance(pose1 , pose2 ):

    puntos1 = []
    puntos2 = []
    for idx, name in key_joints.items():
        print(pose1.joints[name])
        
        if pose1.joints[name] is not None and pose2.joints[name] is not None:
            if pose1.joints[name].visibility > pose1.visibility_threshold and pose2.joints[name].visibility > pose2.visibility_threshold:
                
                
                x1 = pose1.joints[name].x
                y1 = pose1.joints[name].y
                z1 = pose1.joints[name].z
                puntos1.append([x1,y1,z1])

                x2 = pose2.joints[name].x
                y2 = pose2.joints[name].y
                z2 = pose2.joints[name].z
                puntos2.append([x2,y2,z2])
        
    nube_puntos1 = np.array(puntos1)
    nube_puntos2 = np.array(puntos2)

    print("LEN: " + str(len(nube_puntos2)), nube_puntos2)
    
    distancia_wasserstein = wasserstein_distance(nube_puntos1.flatten(), nube_puntos2.flatten())
    print(distancia_wasserstein)
    return distancia_wasserstein



def save_poses(colection,file):
    print("LARGO", len(colection))
    with open(file, 'wb') as serialized:
        pickle.dump(colection, serialized)

def load_poses(file):
    if os.path.exists(file):    
        with open(file, 'rb') as serialized:
            return pickle.load(serialized)
    else:
        return []



class MyApp(App):
    def build(self):
        self.current_pose = None
        self.pose_array = []
        self.saved_poses = 'poses.pkl'
        self.poses = load_poses(self.saved_poses)
        # Diseño principal
        self.layout_principal = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Sub-layouts para las dos secciones superiores
        self.layout_superior = BoxLayout(orientation='horizontal', spacing=10)
        self.layout_izquierdo = BoxLayout(orientation='vertical', spacing=10)
        self.layout_derecho = BoxLayout(orientation='vertical', spacing=10)

        # Placeholder para la imagen de la webcam
        cam_webcam = Camera(play=True)
        self.layout_izquierdo.add_widget(cam_webcam)

        # Video en el cuadro derecho
        video_source = 'video1.mp4'  # Reemplaza con la ruta de tu video
        self.video = Video(source=video_source, state='play', options={'eos': 'loop'}, size_hint=(1, 1))
        self.layout_derecho.add_widget(self.video)

        # Botones en la parte inferior con orientación horizontal
        self.layout_botones = BoxLayout(orientation='horizontal', spacing=10)
        btn_funcion_1 = Button(text='Record Pose', on_press=self.map_pose_gui)
        btn_funcion_2 = Button(text='Save Pose', on_press=self.save_pose)
        btn_funcion_3 = Button(text='Match Pose', on_press=self.match_pose)

        # Agregar widgets a los layouts
        self.layout_superior.add_widget(self.layout_izquierdo)
        self.layout_superior.add_widget(self.layout_derecho)

        self.layout_botones.add_widget(btn_funcion_1)
        self.layout_botones.add_widget(btn_funcion_2)
        self.layout_botones.add_widget(btn_funcion_3)

        self.layout_principal.add_widget(self.layout_superior)
        self.layout_principal.add_widget(self.layout_botones)

        return self.layout_principal

    def map_pose_gui(self, instance):
        name1 = None
        name2 = None
        im1 = im2 = None
        pose1 = None
        if not exp:
            name1 = "img1.png"
            name2 = "img2.png"
            im1, im2 = snap_picture(0,1, name1, name2)
            pose1 = map_pose(im1,im2, name1, name2)
        else:
            name1 = "img1_pose2.png"
            name2 = "img2_pose2.png"
            im1 = cv2.imread(name1)
            im2 = cv2.imread(name2)
            pose1 = map_pose(im1,im2, "test_record_im1.png", "test_record_im1,png")
        
        
        
        pose1.normalize_pose(width, height)
        video_name =  "curr_pose.mp4"
        pose1.generate_video(15, circle_left, circle_right, circle_color,width, height, possible_edges, video_name)
        #video1 = Video(source=video_name, state='play', options={'eos': 'loop'}, size_hint=(1, 1))
        self.current_pose = pose1

        self.layout_derecho.remove_widget(self.video)
        self.video = Video(source=video_name, state='play', options={'eos': 'loop'}, size_hint=(1, 1))
        self.layout_derecho.add_widget(self.video)
       
        


    def save_pose(self, instance):
        self.poses.append(self.current_pose)
        save_poses(self.poses, self.saved_poses)

    def toggle_pose_vision(self, instance):
        print("Función 2 activada")

    def match_pose(self, instance):
        
        print("LARGO", len(self.poses))
        self.current_pose.add_noise(50,150)
        
        mindis = wasserstein_pose_distance(self.poses[0], self.current_pose)
        ind_counter = 0
        match_index = 0

        for pose in self.poses:
            distance = wasserstein_pose_distance(pose, self.current_pose)
            if distance < mindis:
                mindis = distance
                match_index = ind_counter
            ind_counter += 1

        if match_index is not None:
            col1 = ( random.randint(0,255),random.randint(0,255), random.randint(0,255) )
            col2 = ( random.randint(0,255),random.randint(0,255),random.randint(0,255) )

            pose_array = [(self.poses[match_index],col1), (self.current_pose,col2)]
            video_name = "match.mp4"
            print("LLEGO A")
            project_poses(pose_array, 15, circle_left, circle_right, width, height, possible_edges, video_name )

            self.layout_derecho.remove_widget(self.video)
            self.video = Video(source=video_name, state='play', options={'eos': 'loop'}, size_hint=(1, 1))
            self.layout_derecho.add_widget(self.video)



            




    



# FUNCTIONS



# if not exp:
#     name1 = "img1.png"
#     name2 = "img2.png"
#     im1, im2 = snap_picture(0,1, name1, name2)
# else:
#     name1 = "img1_pose2.png"
#     name2 = "img2_pose2.png"
#     im1 = cv2.imread(name1)
#     im2 = cv2.imread(name2)

#     name11 = "img1_pose1.png"
#     name22 = "img2_pose1.png"
#     im11 = cv2.imread(name11)
#     im22 = cv2.imread(name22)


#     name111 = "img1_pose3.png"
#     name222 = "img2_pose3.png"
#     im111 = cv2.imread(name111)
#     im222 = cv2.imread(name222)



#JOINT POSE VIDEO
# for pose in poses:
#     pose_array.append((pose, ( random.randint(0,255),random.randint(0,255),random.randint(0,255) ) ))



# project_poses(pose_array, 15, circle_left, circle_right, width, height, possible_edges)


if __name__ == '__main__':
    MyApp().run()
