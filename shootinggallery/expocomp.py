#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""
Automatic exposure compensation
"""

import logging
logger = logging.getLogger(__name__)
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy

class AutomaticExposureCompensation():



    def __init__(self):
        """TODO: to be defined1. """

        self.startX = 0
        self.startY = 0
        self.endX = 2
        self.endY = 2
        self.image = None
        self.mean = 0
        self.mode = 'normal'


    def set_ref_image(self, image):
        try:
            image.shape
        except:
            import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

            image = np.asarray(image)
            self.mode = 'opencv'
        self.image = image
        self.mean = self.__area_mean(self.image)


    def set_area(self, endX=0, endY=0, startX=2, startY=2):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY
        if self.image is not None: 
            self.mean = self.__area_mean(self.image)

    def __area_mean(self, image):
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        if len(image.shape) == 3:
            mean0 = np.mean(image[
                self.startX:self.endX,
                self.startY:self.endY,
                0])
            mean1 = np.mean(image[
                self.startX:self.endX,
                self.startY:self.endY,
                0])
            mean2 = np.mean(image[
                self.startX:self.endX,
                self.startY:self.endY,
                0])

            mean = np.array([mean0, mean1, mean2])
        else:
            mean = np.mean(image[
                self.startX:self.endX,
                self.startY:self.endY
                ])
        return mean

    def compensate(self, frame):
        mean = self.__area_mean(frame)
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        # print np.max(frame)
        comp = self.mean/mean
        newframe = frame * comp 

        # print np.max(newframe)
        newframe[newframe < 0] = 0
        newframe[newframe > 255] = 255
        newframe[
                self.startX:self.endX,
                self.startY:self.endY
                ] = 0

        return newframe.astype(frame.dtype)



        

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
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)

    import skimage
    import skimage.data
    img = skimage.data.lena()
    img2 = img*0.7
    aec = AutomaticExposureCompensation()
    aec.set_ref_image(img)
    aec.set_area(10, 10)
    aec.compensate(img2)





if __name__ == "__main__":
    main()
