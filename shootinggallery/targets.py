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
import pygame
import cv2

class Target:

    def __init__(self, center, radius, max_score, impath, start=[0 , 0],
            vector=[1, 1], speed=1.0, heading=None): 
        self.center = np.asarray(center)
        self.radius = radius
        self.max_score = max_score
        self.image = pygame.image.load(impath)
        self.score_coeficient = float(max_score) / float(radius)
        self.start = np.asarray(start)
        self.vector = np.asarray(vector)

    def get_score(self, impact_point):
        dist = np.linalg.norm(
            self.center.astype(np.float) - np.asarray(impact_point))
        score = self.max_score - (dist * self.score_coeficient)
        return max(score, 0)

    def draw(self, frame):
        cv2.circle(frame,
                   (self.center[0], self.center[1]),

                   self.radius,
                   (255, 100, 100),
                   2)
        cv2.circle(frame,
                   (self.center[0], self.center[1]),
                   self.radius/10,
                   (255, 100, 100),
                   2)

    def tick(self):
        self.center = self.center + self.vector
# TODO target movement
        pass

class Targets():
    def __init__(self):
        self.targets = []

    def add_from_config(self, config):
        for tg in config:
            target = Target(**tg)
            self.add(target)

    def add(self, target):
        self.targets.append(target)

    def tick(self):
        for tg in self.targets:
            tg.tick()
        pass

    def draw(self, frame):
        for tg in self.targets:
            tg.draw(frame)

    def get_score(self, pt):
        sc = 0.0
        for tg in self.targets:
            sc += tg.get_score(pt)

        return sc

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
