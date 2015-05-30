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

class Target(pygame.sprite.Sprite):

    def __init__(self, center, radius, max_score, impath, start=[0 , 0],
            vector=[1, 1], speed=1.0, heading=None, lifeticks=None):
        pygame.sprite.Sprite.__init__(self)
        self.center = np.asarray(center)
        self.radius = radius
        self.max_score = max_score
        self.image = pygame.image.load(impath)
        self.score_coeficient = float(max_score) / float(radius)
        self.start = np.asarray(start)
        self.vector = np.asarray(vector)
        self.lifeticks = lifeticks
        if impath is "None":
            self.src_image = self.image = pygame.Surface([radius * 2, radius * 2])
        else:
            self.src_image=pygame.image.load(impath)
        self.image = self.src_image
        pygame.draw.circle(self.image, (255, 100,100), self.center, self.radius, 3)
        self.rect = self.image.get_rect()
        self.position = 1.0 * self.center
        self.rect.center = self.position
        self.delete = False

    def get_score(self, impact_point):
        dist = np.linalg.norm(
            self.position + self.center.astype(np.float) - np.asarray(impact_point))
        score = self.max_score - (dist * self.score_coeficient)
        return max(score, 0)

    def _draw(self, frame):
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

    def update(self, deltat):
        self.position = self.position + deltat * 0.001 * self.vector
        self.rect.center = self.position
        if self.lifeticks is not None:
            self.lifeticks -= 1
            if self.lifeticks < 0:
                self.delete = True
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
            if tg.delete:
                self.targets.remove(tg)
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
