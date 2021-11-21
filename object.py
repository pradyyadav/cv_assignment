import os
import shutil
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
from math import sqrt

result_path = './result/'
if os.path.exists(result_path):
    shutil.rmtree(result_path)
os.makedirs(result_path)

for image in os.listdir('./images'):
    print(image)
    img = cv2.imread('./images/' + image)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)





    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 150  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)


    line_dict = {}
    max_len = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            if max_len < max(sqrt((x1-x2)**2+(y1-y2)**2),max_len):
                line_dict['max_line'] = line
                max_len = max(sqrt((x1-x2)**2+(y1-y2)**2),max_len)
                line_dict['max_len'] = max_len


    for x1,y1,x2,y2 in line_dict['max_line']:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imwrite(result_path + '/' + image,lines_edges)
    
    cv2.imshow('image',lines_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()