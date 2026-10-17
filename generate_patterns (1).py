"""Makes the two things you need to print: a chessboard and an ArUco marker.

Run:  python generate_patterns.py
Then print results/chessboard.png and results/marker_id0.png at 100% scale
(no "fit to page"), and stick them on something flat (a book, cardboard).
"""
import os
import cv2
import numpy as np

os.makedirs("results", exist_ok=True)

# 1) Chessboard: 10 x 7 squares  ->  9 x 6 inner corners
SQUARES_X, SQUARES_Y, SQ = 10, 7, 100  # SQ = pixels per square in the image file
board = np.zeros((SQUARES_Y * SQ, SQUARES_X * SQ), np.uint8)
for y in range(SQUARES_Y):
    for x in range(SQUARES_X):
        if (x + y) % 2 == 0:
            board[y * SQ:(y + 1) * SQ, x * SQ:(x + 1) * SQ] = 255
board = cv2.copyMakeBorder(board, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
cv2.imwrite("results/chessboard.png", board)

# 2) ArUco marker (4x4 dictionary, id 0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(aruco_dict, 0, 600)
marker = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
cv2.imwrite("results/marker_id0.png", marker)

print("Saved results/chessboard.png and results/marker_id0.png")
print("After printing, MEASURE with a ruler:")
print("  - one chessboard square (cm)   -> use as --square-cm")
print("  - the black marker side (cm)   -> use as --marker-cm")
