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

def inverted(img):
    inv = pygame.Surface(img.get_rect().size, pygame.SRCALPHA)
    inv.fill((255,255,255,255))
    inv.blit(img, (0,0), None, pygame.BLEND_RGB_SUB)
    return inv

class Target(pygame.sprite.Sprite):
    """
    self.position is the same as self.rect.center
    center is not used any more. in the future there will be connection 
    between self.center and self.position
    """

    def __init__(self, center, radius, max_score, impath, start=[0 , 0],
            vector=[1, 1], speed=1.0, heading=None, lifetime=None, zoom=1.0, 
            invert_intensity=False):
        pygame.sprite.Sprite.__init__(self)
        self.center = (np.asarray(center) * zoom).astype(np.int)
        self.radius = int(radius * zoom)
        self.max_score = max_score
        # self.image = pygame.image.load(impath)
        self.score_coeficient = float(max_score) / float(self.radius)
        self.start = np.asarray(start)
        self.vector = np.asarray(vector)
        self.lifetime = lifetime
        if impath is "None":
            self.src_image = self.image = pygame.Surface([radius * 2, radius * 2])
        else:
            self.src_image=pygame.image.load(impath)
        self.image = pygame.transform.rotozoom(self.src_image, 0, zoom)
        if invert_intensity:
            self.image = inverted(self.image)
        # pygame.draw.circle(self.image, (255, 100,100), self.center, self.radius, 3)
        self.rect = self.image.get_rect()
        self.position = 1.0 * np.asarray(start)
        self.rect.center = self.position
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        pygame.draw.circle(self.image, (255, 100,100), self.center, self.radius, 3)
        self.delete = False

    def get_score(self, impact_point):

        dist = np.linalg.norm(
                np.array([self.rect.left, self.rect.top]) + self.center
            # self.position.astype(np.float)
            - np.asarray(impact_point))
        # score = self.max_score - (dist * self.score_coeficient)
        score = (1.0 - dist/self.radius)
        score = self.max_score * max(score, 0)

        return score

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
        # self.rect = self.rect.move(self.vector * deltat * 0.01)
        # print 've' , self.vector
        # print 'rc' , self.rect.center
        # print 'po' , self.position
        self.position = self.position + deltat * 0.001 * self.vector
        self.rect.center = self.position
        if self.lifetime is not None:
            self.lifetime -= (deltat * 0.001)
            if self.lifetime < 0:
                self.delete = True
                self.kill()
# TODO target movement
        pass

class TargetGenerator():
    def __init__(self, target_generator_config): 
        selg.tgg_config = target_generator_config
        self.time = 0.0
        self.target_list = None

    def update(self, deltat):
        self.time -= deltat
        if self.time < 0:
            self.time = 10.0

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
