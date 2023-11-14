import math
import cv2
import random
class Point:
    
    def __init__(self, x_, y_, z_):
        self.x = x_
        self.y = y_
        self.z = z_

class Joint:
    def __init__(self, x_, y_, z_, visibility_, id_: str):
        self.x = x_
        self.y = y_
        self.z = z_
        self.visibility = visibility_
        self.id = id_

class Pose:
    
    def __init__(self, l_shoulder_ : (float,float, float), r_shoulder_ : (float,float, float)):
        self.l_shoulder = l_shoulder_
        self.r_shoulder = r_shoulder_
        self.key_length = distance_2(self.l_shoulder,self.r_shoulder)

    def __init__(self, key_joints, visibility_threshold):
        self.img_array = []
        self.visibility_threshold = visibility_threshold
        self.joints = {
        "LEFT_SHOULDER":None,
        "RIGHT_SHOULDER":None,
        "LEFT_ELBOW":None,
        "RIGHT_ELBOW":None,
        "LEFT_WRIST":None,
        "RIGHT_WRIST":None,
        "LEFT_HIP":None,
        "RIGHT_HIP":None,
        "LEFT_KNEE":None,
        "RIGHT_KNEE":None,
        "LEFT_ANKLE":None,
        "RIGHT_ANKLE":None,
        "LEFT_HEEL":None,
        "RIGHT_HEEL":None,
        }
        self.key_joints = key_joints
    



    def add_joint(self, joint_ : Joint):
        joint_name = self.key_joints[joint_.id]
        self.joints[joint_name] = joint_

    def find_center(self):

        x1 = self.joints["LEFT_SHOULDER"].x
        x2 = self.joints["RIGHT_SHOULDER"].x
        x3 = self.joints["LEFT_HIP"].x
        x4 = self.joints["RIGHT_HIP"].x

        y1 = self.joints["LEFT_SHOULDER"].y
        y2 = self.joints["RIGHT_SHOULDER"].y
        y3 = self.joints["LEFT_HIP"].y
        y4 = self.joints["RIGHT_HIP"].y

        center_x = (x1 + x2 + x3 + x4) / 4
        center_y = (y1 + y2 + y3 + y4) / 4

        return center_x, center_y



    def normalize_pose(self, width, height):
        

        #Alineacion de arista hombro_izq - hombro_der
        z_tolerance = 10

        main_edge_goal = 500

        for angle in range(360):
            self.key_point = Point(int(width/2), int(height/2), 0)
            alt_joints = dict(self.joints)
            rad_angle =  math.radians(angle)
            
            for name, joint in alt_joints.items():
                if joint is not None:
                    new_x= self.key_point.x + (joint.x - self.key_point.x) * math.cos(rad_angle) - (joint.z - self.key_point.z) * math.sin(rad_angle)
                    new_z = self.key_point.z + (joint.x- self.key_point.x) * math.sin(rad_angle) + (joint.z - self.key_point.z) * math.cos(rad_angle)
                    new_joint = Joint(new_x, joint.y, new_z, joint.visibility, joint.id)
                    alt_joints[name] = new_joint 
            
            if abs(alt_joints["LEFT_SHOULDER"].z - alt_joints["RIGHT_SHOULDER"].z) <= z_tolerance and alt_joints["LEFT_SHOULDER"].x < alt_joints["RIGHT_SHOULDER"].x:
                 break
        self.joints = alt_joints

        

        c_x, c_y = self.find_center()
        key_y = height / 2
        key_z = 0 
        key_x = width/2
        key_z = (self.joints["LEFT_SHOULDER"].z + self.joints["RIGHT_SHOULDER"].z)/2

        for name, joint in self.joints.items():
            if joint is not None:
                if joint.visibility > self.visibility_threshold:
                    joint.z = joint.z - key_z
                    joint.x = joint.x + (key_x - joint.x) + (joint.x - c_x)
                    joint.y = joint.y + (key_y - joint.y) + (joint.y - c_y)


        main_edge_real = self.joint_distance(self.joints["LEFT_SHOULDER"], self.joints["RIGHT_SHOULDER"])

        zoom_factor = main_edge_goal / main_edge_real


        for name, joint in self.joints.items():
            if joint is not None:
                if joint.visibility > self.visibility_threshold:
                    joint.z = joint.z * zoom_factor
                    joint.x = joint.x * zoom_factor
                    joint.y = joint.y * zoom_factor

        print(self.joint_distance(self.joints["LEFT_SHOULDER"], self.joints["RIGHT_SHOULDER"]))


    def add_circles_independent(self, image, joints, circle_radius, circle_left, circle_right, circle_color):
    
        for joint_name,joint in joints.items():
            if joint is not None:
                if joint.visibility > self.visibility_threshold:
                    if joint_name == "LEFT_SHOULDER":
                        cv2.circle(image, (int(joint.x),int(joint.y)), 15,  circle_left  , -1) 
                    elif joint_name == "RIGHT_SHOULDER":
                        cv2.circle(image, (int(joint.x),int(joint.y)), circle_radius,  circle_right  , -1)
                    else:
                        cv2.circle(image, (int(joint.x),int(joint.y)), circle_radius,  circle_color  , -1)
                            
    def project_pose(self, angle, circle_radius, circle_left, circle_right, circle_color, width, height, possible_edges, center = True):
        cx, cy = self.find_center()
        self.key_point = Point(cx,cy,0)
        alt_joints = dict(self.joints)
        rad_angle =  math.radians(angle)
        blank1 = cv2.imread("blank.jpg")
        blank_rgb1 = cv2.cvtColor(blank1,cv2.COLOR_BGR2RGB)
        
        if center:
            for name, joint in alt_joints.items():
                if joint is not None:
                    new_x= self.key_point.x + (joint.x - self.key_point.x) * math.cos(rad_angle) - (joint.z - self.key_point.z) * math.sin(rad_angle)
                    new_z = self.key_point.z + (joint.x- self.key_point.x) * math.sin(rad_angle) + (joint.z - self.key_point.z) * math.cos(rad_angle)
                    new_joint = Joint(new_x, joint.y, new_z, joint.visibility, joint.id)
                    alt_joints[name] = new_joint
        
        else:
            for name, joint in alt_joints.items():
                if joint is not None:
                    new_x =  joint.x  * math.cos(rad_angle) - joint.z * math.sin(rad_angle)
                    new_z = joint.x * math.sin(rad_angle) + joint.z * math.cos(rad_angle)
                    new_joint = Joint(new_x, joint.y, new_z, joint.visibility, joint.id)
                    alt_joints[name] = new_joint

        self.add_circles_independent(blank_rgb1, alt_joints, circle_radius, circle_left, circle_right, circle_color)
        

        for edge in possible_edges:
            if self.joints[edge[0]] is not None and self.joints[edge[1]] is not None:
                if self.joints[edge[0]].visibility > self.visibility_threshold and self.joints[edge[1]].visibility > self.visibility_threshold:
                    cv2.line(blank_rgb1, (int(alt_joints[edge[0]].x) , int(alt_joints[edge[0]].y)), (int(alt_joints[edge[1]].x ), int(alt_joints[edge[1]].y)), circle_color, 10)

        

                
        '''
        cv2.line(blank_rgb1, (int(alt_joints["LEFT_SHOULDER"].x) , int(alt_joints["LEFT_SHOULDER"].y)), (int(alt_joints["RIGHT_SHOULDER"].x ), int(alt_joints["RIGHT_SHOULDER"].y)), circle_color, 10)
        
        cv2.line(blank_rgb1, (int(alt_joints["LEFT_SHOULDER"].x) , int(alt_joints["LEFT_SHOULDER"].y)), (int(alt_joints["LEFT_ELBOW"].x ), int(alt_joints["LEFT_ELBOW"].y)), circle_color, 10)
        cv2.line(blank_rgb1, (int(alt_joints["RIGHT_SHOULDER"].x) , int(alt_joints["RIGHT_SHOULDER"].y)), (int(alt_joints["RIGHT_ELBOW"].x ), int(alt_joints["RIGHT_ELBOW"].y)), circle_color, 10)

        cv2.line(blank_rgb1, (int(alt_joints["RIGHT_ELBOW"].x) , int(alt_joints["RIGHT_ELBOW"].y)), (int(alt_joints["RIGHT_WRIST"].x ), int(alt_joints["RIGHT_WRIST"].y)), circle_color, 10)
        cv2.line(blank_rgb1, (int(alt_joints["LEFT_ELBOW"].x) , int(alt_joints["LEFT_ELBOW"].y)), (int(alt_joints["LEFT_WRIST"].x ), int(alt_joints["LEFT_WRIST"].y)), circle_color, 10)

        
        cv2.line(blank_rgb1, (int(alt_joints["LEFT_SHOULDER"].x) , int(alt_joints["LEFT_SHOULDER"].y)), (int(alt_joints["RIGHT_HIP"].x ), int(alt_joints["RIGHT_HIP"].y)), circle_color, 10)
        cv2.line(blank_rgb1, (int(alt_joints["RIGHT_SHOULDER"].x) , int(alt_joints["RIGHT_SHOULDER"].y)), (int(alt_joints["LEFT_HIP"].x ), int(alt_joints["LEFT_HIP"].y)), circle_color, 10)
        '''

        self.img_array.append(blank_rgb1)
        #plt.imshow(blank_rgb1)    
        
    def generate_video(self, circle_radius, circle_left, circle_right, circle_color, width, height, possible_edges):
        self.img_array = []

        for i in range(0,180):
            self.project_pose(i,circle_radius, circle_left, circle_right, circle_color,width, height,possible_edges)

        codec = cv2.VideoWriter_fourcc(*'mp4v')     
        out = cv2.VideoWriter('video1.mp4',codec, 36, (width,height))

        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
            #cv2.imwrite("rotation/" + str(i) + "photo.png" , self.img_array[i])

        out.release()


    def add_circles(self, image,  circle_radius, circle_left, circle_right, circle_color ):
        
        for joint_name,joint in self.joints.items():
            if joint.visibility > self.visibility_threshold:
                if joint_name == "LEFT_SHOULDER":
                    cv2.circle(image, (int(joint.x),int(joint.y)), circle_radius,  circle_left  , -1) 
                elif joint_name == "RIGHT_SHOULDER":
                    cv2.circle(image, (int(joint.x),int(joint.y)), circle_radius,  circle_right  , -1)
                else:
                 cv2.circle(image, (int(joint.x),int(joint.y)), circle_radius,  circle_color  , -1)

    def joint_distance(self, joint1 : Joint, joint2 : Joint):
        x1 = joint1.x
        y1 = joint1.y
        z1 = joint1.z
        x2 = joint2.x
        y2 = joint2.y
        z2 = joint2.z
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return distance


    def add_noise(self, minval, maxval):

        for joint_name, joint in self.joints.items():
            if joint is not None:
                if joint.visibility > self.visibility_threshold:
                    noise = random.randint(minval, maxval)
                    joint.z += noise
                    joint.x += noise
                    joint.y += noise






    def normalize(self):

        if self.joints['LEFT_SHOULDER'] is not None and self.joints["RIGHT_SHOULDER"] is not None:
            if self.joints['LEFT_SHOULDER'].visibility > self.visibility_threshold and self.joints["RIGHT_SHOULDER"].visibility > visibility_threshold:
                key_distance = 10
                real_distance = self.joint_distance(self.joints['LEFT_SHOULDER'], self.joints['RIGHT_SHOULDER'])


                   