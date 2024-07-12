import cv2
import numpy as np
import json
import os
import sys
import time
import pygame

pygame.init()

dirname = os.path.dirname(__file__)

with open(dirname + '/arducam.json', 'r') as f:
    calibration_data = json.load(f)

# Vision setup
cap = cv2.VideoCapture(1)
tag_size_meters = 0.03
img_alpha = 1.5
img_beta = 0
camera_matrix = np.array(calibration_data["camera_matrix"]["data"]).reshape(3, 3)
dist_coeffs = np.array(calibration_data["distortion_coefficients"]["data"]).reshape(5, 1)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
aruco_params = cv2.aruco.DetectorParameters()

# Renderer
displays = pygame.display.get_num_displays()
display_index = displays - 1
desktop = pygame.display.get_desktop_sizes()[display_index]
window = pygame.display.set_mode(desktop, display=display_index, flags=pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption('Holoscreen')
window.fill((0, 0, 0))

# tag10
tag = pygame.image.load(dirname + '/tag_10.png')
tag = pygame.transform.scale(tag, (500, 500))
window.blit(tag, (window.get_width() / 2 - tag.get_width() / 2, window.get_height() / 2 - tag.get_height() / 2))

# tag0
tag = pygame.image.load(dirname + '/tag_0.png')
tag = pygame.transform.scale(tag, (200, 200))
window.blit(tag, (0, 0))

# tag1
tag = pygame.image.load(dirname + '/tag_1.png')
tag = pygame.transform.scale(tag, (200, 200))
window.blit(tag, (window.get_width() - tag.get_width(), 0))

# tag2
tag = pygame.image.load(dirname + '/tag_2.png')
tag = pygame.transform.scale(tag, (200, 200))
window.blit(tag, (0, window.get_height() - tag.get_height()))

# tag3
tag = pygame.image.load(dirname + '/tag_3.png')
tag = pygame.transform.scale(tag, (200, 200))
window.blit(tag, (window.get_width() - tag.get_width(), window.get_height() - tag.get_height()))

# Create a surface to draw on with distortion
distorted_surface = pygame.Surface((window.get_width(), window.get_height()))

# add a text to the surface
font = pygame.font.Font(None, 36)
text = font.render('Hello, World!', True, (255, 255, 255))
distorted_surface.blit(text, (window.get_width() / 2 - text.get_width() / 2, window.get_height() / 2 - text.get_height() / 2))

def apply_transformation(surface, rvec, tvec, camera_matrix, dist_coeffs):
    height, width = surface.get_height(), surface.get_width()
    pixels = pygame.surfarray.array3d(surface)
    transformed_pixels = np.zeros_like(pixels)

    # Create the mesh grid of the image
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(xv)
    points_2d = np.stack([xv, yv, ones], axis=-1).reshape(-1, 3).T

    # Undistort the points
    points_2d_undistorted = cv2.undistortPoints(np.expand_dims(points_2d[:2].T, axis=1), camera_matrix, dist_coeffs, P=camera_matrix).reshape(-1, 2).T

    # Convert the 2D points to 3D by adding a z-coordinate (which can be zero if the image is flat)
    points_3d = np.vstack((points_2d_undistorted, np.zeros((1, points_2d_undistorted.shape[1]))))

    # Apply the rotation and translation
    rmat, _ = cv2.Rodrigues(rvec)
    transformed_points_3d = rmat @ points_3d + tvec

    # Project the 3D points back to 2D
    projected_points_2d = camera_matrix @ transformed_points_3d
    projected_points_2d /= projected_points_2d[2]

    # Create the transformed pixel array
    for i in range(points_2d.shape[1]):
        x, y = points_2d[:2, i].astype(int)
        x_proj, y_proj = projected_points_2d[:2, i].astype(int)

        if 0 <= x_proj < width and 0 <= y_proj < height:
            transformed_pixels[x_proj, y_proj] = pixels[x, y]

    # Create a new surface from the transformed pixels
    transformed_surface = pygame.surfarray.make_surface(transformed_pixels)
    return transformed_surface

def update_loop():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=img_alpha, beta=img_beta)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, tag_size_meters, camera_matrix, dist_coeffs)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    if ids is not None:
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
        window.blit(apply_transformation(distorted_surface, rvecs[0], tvecs[0], camera_matrix, dist_coeffs), (0, 0))
    

    # Display the frame
    cv2.imshow('frame', frame)
    

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the vision
    update_loop()

    # Update the display
    pygame.display.flip()

    time.sleep(0.016)

pygame.quit()
sys.exit()