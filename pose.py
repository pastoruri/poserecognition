import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


with mp_pose.Pose(static_image_mode = True) as pose:

    image = cv2.imread("image.jpg")
    cv2.waitKey(0)

cv2.destroyAllWindows()
