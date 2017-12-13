#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN based Illegal Parking Detection Algorithm
# Licensed under The MIT License [see LICENSE for details]
# Written by Pratesh Kumar, based on code from EnderNewton's Faster RCNN
# --------------------------------------------------------

# Resources
# http://answers.opencv.org/question/5163/how-to-use-callback-to-draw-a-rectangle-in-a-video/
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html#drawing-functions
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html#mouse-handling
#


"""
Main Script for Illegal Parking Detection User Application
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, cv2

video_path = "/home/vca_ann/dataset/gate_2_BDIP_best1.mp4"

ip_video = cv2.VideoCapture(video_path)

ix,iy = -1,-1
callback = False

max_vertices = 8

pts = [0,0] * max_vertices
count = 0
pts_poly1 = []
pts_poly2 = []
pts_poly3 = []
pts_poly4 = []

car_centroid = (100,100)

#Take the first frame
ret, ip_frame = ip_video.read()

def draw_polygon(event,x,y,flags,params):
	global pts, pts_poly1, count, callback, pts_poly1
	if event == cv2.EVENT_LBUTTONDOWN:
		if(count < max_vertices):
			pts[count*2] = x
			pts[count*2+1] = y
			count = count + 1
		if (count == max_vertices):
			print("Max reached")
			callback = True
			pts_poly1 = np.array([pts], np.int32)
			pts_poly1 = pts_poly1.reshape((-1,1,2))
	elif event == cv2.EVENT_MBUTTONDOWN: #click middle button of mouse if you intend to end the polygon drawing before reaching the max_vertices
		while (count < max_vertices):			
			pts[count*2] = x
			pts[count*2+1] = y
			count = count + 1
			print(pts)
		if (count == max_vertices):
			print("Polygon drawing is complete")
			callback = True
			pts_poly1 = np.array([pts], np.int32)
			pts_poly1 = pts_poly1.reshape((-1,1,2))
			 
				

cv2.namedWindow('IPD_Final_Window')
cv2.imshow('IPD_Final_Window',ip_frame)
cv2.setMouseCallback('IPD_Final_Window',draw_polygon)


while(True):
	if(callback):
			#capture frame by frame
			ret, ip_frame = ip_video.read()				
			cv2.polylines(ip_frame,[pts_poly1],True,(0,255,255))
			cv2.imshow('IPD_Final_Window', ip_frame)
			if (cv2.pointPolygonTest(pts_poly1,car_centroid,False) > 0):
				print("Point inside")
	if cv2.waitKey(1) & 0xFF == ord('q'):			
		break		




"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, cv2

video_path = "/home/vca_ann/dataset/gate_2_BDIP_best1.mp4"

ip_video = cv2.VideoCapture(video_path)

#first frame
ret, ip_frame = ip_video.read() 

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))

drawing = False # true if mouse is pressed
ix,iy= -1,-1
tx,ty= -1,-1
callback = False

#mouse call-back function
def draw_polygon(event,x,y,flags,params):
	global ix,iy,drawing,ip_frame,callback,tx,ty
	
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		print("L BUTTON DOWN")
		ix,iy = x,y

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			print("L BUTTON MOVE")
			tx,ty=x,y
			cv2.line(ip_frame,(ix,iy),(tx,ty),(255,0,0),2)
			cv2.imshow('IPD_Final_Window', ip_frame)

	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		print("L BUTTON UP")
		tx,ty = x,y
		cv2.line(ip_frame,(ix,ix),(tx,ty),(255,255,0),2)
		cv2.imshow('IPD_Final_Window', ip_frame)
		callback = True
		print("L BUTTON UP AFTERWRDS")
		print("%d %d %d %d \n",ix,iy,tx,ty)
		cv2.waitKey(10)

cv2.namedWindow('IPD_Final_Window')
cv2.imshow('IPD_Final_Window',ip_frame)
cv2.setMouseCallback('IPD_Final_Window',draw_polygon)

if __name__ == '__main__':
	#global callback
	while(True):
		if(callback):
			print("TRUE CALLBACK")
			#capture frame by frame
			ret, ip_frame = ip_video.read()
		
		
			cv2.polylines(ip_frame,[pts],True,(0,255,255))
			cv2.imshow('IPD_Final_Window', ip_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("%d %d %d %d \n",ix,iy,tx,ty)
			break


#Release the capture
ip_video.release()
cv2.destroyAllWindows()

"""
    
