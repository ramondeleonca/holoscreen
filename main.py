import pygame
import os
import json
import cv2
import json
import numpy as np

BIG_TAG_SIZE = 500
SMALL_TAG_SIZE = 200

dirname = os.path.dirname(__file__)

with open(dirname + '/lifecam.json', 'r') as f:
    calibration_data = json.load(f)

# Vision setup
cap = cv2.VideoCapture(1)
tag_size = 0.3
img_alpha = 1.5
img_beta = 0
camera_matrix = np.array(calibration_data["camera_matrix"]["data"]).reshape(3, 3)
dist_coeffs = np.array(calibration_data["distortion_coefficients"]["data"]).reshape(5, 1)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()

pygame.init()

displays = pygame.display.get_num_displays()
display_index = displays - 1
desktop = pygame.display.get_desktop_sizes()[display_index]
window = pygame.display.set_mode(desktop, display=display_index, flags=pygame.DOUBLEBUF)
pygame.display.set_caption('Holoscreen')
window.fill((0, 0, 0))

# tag10
tag = pygame.image.load(dirname + '/tag_10.png')
tag = pygame.transform.scale(tag, (BIG_TAG_SIZE, BIG_TAG_SIZE))
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

def vision_update_loop():
    # Read camera
    ret, frame = cap.read()

    # Skip if no frame
    if not ret:
        return
    
    # Apply frame effects
    frame = cv2.convertScaleAbs(frame, alpha=img_alpha, beta=img_beta)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        # Resolve markers
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, tag_size, camera_matrix, dist_coeffs)

        # Draw markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # Display frame
    cv2.imshow('frame', frame)

    # Display another frame that crops the tag
    if ids is not None:
        for i in range(len(ids)):
            if ids[i] == 10:
                x, y, w, h = cv2.boundingRect(corners[i])
                cropped = frame[y:y+h, x:x+w]

                # warp the image to correct the perspective
                h, w = cropped.shape[:2]

                # pts1 is the corners of the tag and pts2 is the tag size
                pts1 = np.float32(corners[i][0] - [x, y])
                pts2 = np.float32([[0, 0], [BIG_TAG_SIZE, 0], [BIG_TAG_SIZE, BIG_TAG_SIZE], [0, BIG_TAG_SIZE]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                cropped = cv2.warpPerspective(cropped, M, (BIG_TAG_SIZE, BIG_TAG_SIZE))

                cropped = cv2.flip(cropped, -1)

                cv2.imshow('cropped'+str(ids[i]), cropped)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    vision_update_loop()
    
    pygame.display.flip()

pygame.quit()