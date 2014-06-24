'''
Created on 21.5.2014

@author: Michal
'''

""" automatic_calibration"""
image_projector = "obP2.jpg"
image_kinect = "obK8.jpg"
min_count_of_matches = 10

""" sledovani """
kinect_server_adress = "ws://192.168.1.100:9002"
LOOP_TIME = 0.15
window_width = 1500
window_height = 900
background = "black.png"
red_dot = "red_dot.png"
green_dot = "green_dot.png"

im_folder = "im/"
im_directory = "obr/"
# MODE = "normal"
MODE = "dem"
calibration_mode = "ransac"


point_torso = "no"
point_neck = "no"
point_head = "no"
image = "yes"

spacebar_to_toggle_images = "yes"
spacebar_to_toggle_directory = "no"









