import tkinter
import cv2
import numpy as np
import json
import os
from PIL import ImageTk, Image
from screeninfo import get_monitors

dirname = os.path.dirname(__file__)

with open(dirname + '/camera.json', 'r') as f:
    calibration_data = json.load(f)

cap = cv2.VideoCapture(1)
tag_size_meters = 0.03
img_alpha = 1.5
img_beta = 0
camera_matrix = np.array(calibration_data["camera_matrix"]["data"]).reshape(3, 3)
dist_coeffs = np.array(calibration_data["distortion_coefficients"]["data"]).reshape(5, 1)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
aruco_params = cv2.aruco.DetectorParameters()

screen = get_monitors()[-1]
root = tkinter.Tk()
root.geometry(f"{screen.width}x{screen.height}+{screen.x}+{screen.y}")
root.configure(bg='#111111')
root.overrideredirect(True)
root.attributes('-topmost', True)

img = Image.open(os.path.join(dirname, 'tag_10.png'))
img = img.resize((300, 300), Image.NEAREST)
img = ImageTk.PhotoImage(img)
panel = tkinter.Label(root, image=img)
panel.pack(expand=True)

def update_loop():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=img_alpha, beta=img_beta)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, tag_size_meters, camera_matrix, dist_coeffs)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    for i in range(len(ids)):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
    print(corners, ids)
    cv2.imshow('frame', frame)
    root.after(1, update_loop)

root.after(1, update_loop)

root.mainloop()