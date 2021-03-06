#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""

"""

import logging
logger = logging.getLogger(__name__)
import argparse
import numpy as np
import cv2
import cStringIO
import scipy
import scipy.misc
import cv

def toggle_fullscreen():
    screen = pygame.display.get_surface()
    tmp = screen.convert()
    caption = pygame.display.get_caption()
    cursor = pygame.mouse.get_cursor()  # Duoas 16-04-2007 
    
    w,h = screen.get_width(),screen.get_height()
    flags = screen.get_flags()
    bits = screen.get_bitsize()
    
    pygame.display.quit()
    pygame.display.init()
    
    screen = pygame.display.set_mode((w,h),flags^pygame.locals.FULLSCREEN,bits)
    screen.blit(tmp,(0,0))
    pygame.display.set_caption(*caption)
 
    pygame.key.set_mods(0) #HACK: work-a-round for a SDL bug??
 
    pygame.mouse.set_cursor( *cursor )  # Duoas 16-04-2007
    
    return screen

def np2surf(pixels, transpose=True):
    import pygame
    import pygame.surfarray

    if transpose:
        # pixels = cv2.transpose(pixels)
        if len(pixels.shape) == 3:
            pixels = np.transpose(pixels, axes=[1, 0, 2])
        else:
            pixels = np.transpose(pixels, axes=[1, 0])
    try:
        surf = pygame.surfarray.make_surface(pixels)
    except IndexError:
        if len(pixels.shape) == 2:
            sh = pixels.shape
            px = np.zeros([sh[0], sh[1], 3], dtype=pixels.dtype)

            px[:, :, 0] = pixels[:, :]
            px[:, :, 1] = pixels[:, :]
            px[:, :, 2] = pixels[:, :]
            pixels = px
        (width, height, colours) = pixels.shape
        surf = pygame.Surface((width, height))
        # surf = pygame.display.set_mode((width, height))
        pygame.surfarray.blit_array(surf, pixels)
    return surf

class FrameGetter():
    """
    Can be used for reading image or video from file or url
    """
    def __init__(self, video_source=0, rot90=False, color='RGB', resolution=None):
        """
        resolution works only for opencv
        """
        resolution
        self.rot90 = rot90
        self.color = color
        self.video_source = video_source

        try:
            file = cStringIO.StringIO(urllib.urlopen(video_source).read())
            img = scipy.misc.imread(file)
            self.useopencv = False
        except:
            self.useopencv = True 

        if self.useopencv:
            self.cap = cv2.VideoCapture(self.video_source)
            if resolution is not None:
                self.cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, resolution[0])
                self.cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, resolution[1])
            if True:
                print "FPS ", self.cap.get(cv.CV_CAP_PROP_FPS)
                self.cap.set(cv.CV_CAP_PROP_FPS, 10)

    def read(self):
        if self.useopencv:
            res, frame = self.cap.read()
            if self.color is "RGB":
                frame = frame[:,:, ::-1]
            if self.rot90:
                frame = cv2.transpose(frame)
        else:
            file = cStringIO.StringIO(urllib.urlopen(video_source).read())
            frame = scipy.misc.imread(file)
            res = True
            if self.rot90:
                frame = np.rot90(frame)

        return res, frame

    def release(self):
        if self.useopencv:
            self.cap.release()

def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)



if __name__ == "__main__":
    main()
