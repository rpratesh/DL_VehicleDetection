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
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
# 


"""
Main Script for Illegal Parking Detection User Application

1) Mutiple IPZs
2) Time elapsed based alarm raising
3) Combining traditinal CV methods and DL ('coz DL is not able to detect the Illegal Parking if the car is not completly in the camera view
										and imporve the speed of detection by using a hybrid algorithm)
4) More UI features
5) IoU area based alarm instead of Rectangle centre based alarm
6) Combining inferences from different DL architectures - say pascal and mio_tcd based vgg16 and res101s
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',  # always index 0
           'bicycle',
           'bus', 'car',                     
           'motorbike', 'person')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def demo(sess, net, ip_image):
	"""Detect object classes in an image using pre-computed object proposals."""

	# Detect all object classes and regress object bounds
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(sess, net, ip_image)
	timer.toc()
	#print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
	all_class_dets = []
	# Visualize detections for each class
	CONF_THRESH = 0.8
	NMS_THRESH = 0.3
	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		all_class_dets.append(dets)
	return all_class_dets
	#return dets

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
		                choices=NETS.keys(), default='res101')
	parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
		                choices=DATASETS.keys(), default='pascal_voc_0712')
	args = parser.parse_args()

	return args


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
			pts[count*2] = pts[0]
			pts[count*2+1] = pts[1]
			count = count + 1
			print(pts)
		if (count == max_vertices):
			print("Polygon drawing is complete")
			callback = True
			pts_poly1 = np.array([pts], np.int32)
			pts_poly1 = pts_poly1.reshape((-1,1,2))			 				



if __name__ == '__main__':

	video_path = "/home/vca_ann/dataset/parking_1920_part_2.mp4"

	ip_video = cv2.VideoCapture(video_path)

	callback = False

	max_vertices = 8

	font = cv2.FONT_HERSHEY_SIMPLEX

	pts = [0,0] * max_vertices
	count = 0
	pts_poly1 = []
	pts_poly2 = []
	pts_poly3 = []
	pts_poly4 = []

	ipd_detected = 0

	#Take the first frame
	ret, ip_frame = ip_video.read()

	cv2.namedWindow('IPD_Final_Window')
	cv2.imshow('IPD_Final_Window',ip_frame)
	cv2.setMouseCallback('IPD_Final_Window',draw_polygon)

	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	args = parse_args()

	# model path
	demonet = args.demo_net
	dataset = args.dataset
	tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
		                      NETS[demonet][0])


	if not os.path.isfile(tfmodel + '.meta'):
		raise IOError(('{:s} not found.\nDid you download the proper networks from '
		               'our server and place them properly?').format(tfmodel + '.meta'))

	# set config
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth=True

	# init session
	sess = tf.Session(config=tfconfig)
	# load network
	if demonet == 'vgg16':
		net = vgg16()
	elif demonet == 'res101':
		net = resnetv1(num_layers=101)
	else:
		raise NotImplementedError
	net.create_architecture("TEST", 6,
		                  tag='default', anchor_scales=[8, 16, 32])
	saver = tf.train.Saver()
	saver.restore(sess, tfmodel)

	print('Loaded network {:s}'.format(tfmodel))

	while(True):
		if(callback):
			#capture frame by frame
			ret, ip_frame = ip_video.read()							
			all_rects = demo(sess, net, ip_frame) # all_rects will cotnain the bboxes of all the classes along with scores
			
			for class_ind in range(len(CLASSES)-2):	 #-2 'coz we;ve to avoid detecting background and humans - bg was already filtered, humans we'r filtering here
				rects = all_rects[class_ind]
				#print(rects)			

				inds = np.where(rects[:, -1] >= 0.8)[0] #0.8 is threshold .. can change it
				if len(inds) == 0:
					#print("No rects")
					cv2.polylines(ip_frame,[pts_poly1],True,(0,255,255))
					cv2.imshow('IPD_Final_Window', ip_frame) #even if there'r no detections, the video should run
					if cv2.waitKey(1) & 0xFF == ord('q'):			
						break
					continue

				cv2.polylines(ip_frame,[pts_poly1],True,(0,255,255))
			
				for i in inds:
					bbox = rects[i, :4]
					cv2.rectangle(ip_frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
					cv2.putText(ip_frame,str(CLASSES[class_ind+1]),(bbox[0],bbox[1]), font, 0.5,(0,255,255),1,cv2.LINE_AA)
					if (cv2.pointPolygonTest(pts_poly1,(bbox[0]+int(bbox[2]-bbox[0])/2,bbox[1]+int(bbox[3]-bbox[1])/2),False) > 0):   #Point polygon test --> tests if the rect's centre lies inside the polygon
						ipd_detected = 1						
						
					#break
			
			if ipd_detected == 1:
				#print("Illegal Parking detected in Zone 1")
				#cv2.putText(ip_frame,'IPD',(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
				cv2.polylines(ip_frame,[pts_poly1],True,(0,0,255))
			ipd_detected = 0
			cv2.imshow('IPD_Final_Window', ip_frame)

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
    
