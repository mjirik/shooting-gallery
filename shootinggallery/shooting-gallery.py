#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hra střelnice
"""

# import pygame
import argparse
import ConfigParser
import sys
import json
import cv2
import numpy as np

import blob_detection as bd


class Target:

    def __init__(self, center, radius, max_score):
        self.center = np.asarray(center)
        self.radius = radius
        self.max_score = max_score
        self.score_coeficient = float(max_score) / float(radius)

    def get_score(self, impact_point):
        dist = np.linalg.norm(
            self.center.astype(np.float) - np.asarray(impact_point))
        score = self.max_score - (dist * self.score_coeficient)
        return max(score, 0)

    def draw_target(self, frame):
        cv2.circle(frame,
                   (self.center[0], self.center[1]),
                   self.radius, 200, 0)
        cv2.circle(frame,
                   (self.center[0], self.center[1]),
                   self.radius/10, 200, 0)


class ShootingGallery():

    def __init__(self):
        """
        Inicializační funkce. Volá se jen jednou na začátku.
        """
# create video capture
        self.cap = cv2.VideoCapture(0)
        self.target = Target([300, 300], 200, 10)
        self.status_text = ""
        pass

    def tick(self):
        """
        Tato funkce se vykonává opakovaně
        """
        # read the frames
        _, frame = self.cap.read()

        keypoints = bd.red_dot_detection(frame)

        # smooth it
        for i, keypoint in enumerate(keypoints):
            cx = int(keypoint.pt[0])
            cy = int(keypoint.pt[1])
            cv2.circle(frame, (cx, cy), 5 + i, 255, -1)
            # cv2.circle(frame,keypoints[0].pt,5,255,-1)
            # print self.target.get_score([cx,cy])

            if i == 0:
                self.status_text = str(self.target.get_score([cx, cy]))
                print self.status_text
        # Show it, if key pressed is 'Esc', exit the loop

        self.print_status(frame)
        self.target.draw_target(frame)
        cv2.imshow('frame', frame)
        # cv2.imshow('thresh',inv_r_channel)
        if cv2.waitKey(33) == 27:
            return False
        return True

    def run(self):
        """
        funkce opakovaně volá funkci tick
        """

        while 1:
            # casovac.tick(20)
            # Ošetření vstupních událostí
            # for udalost in pygame.event.get():
            #     if udalost.type == pygame.locals.QUIT:
            #         return
            #     elif (udalost.type == pygame.locals.KEYDOWN and
            #           udalost.key == pygame.locals.K_ESCAPE):
            #         return
            # casovac = pygame.time.Clock()

            self.tick()

        self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.cap.release()

    def print_status(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            self.status_text,
            (100, 100), font, 4, (255, 255, 255))  # ,2,cv2.LINE_AA)


def main():
    args = get_params()
    print vars(args)
    # convert to ints
    print json.loads(args.target_center)
    sh = ShootingGallery()
    sh.run()


def get_params(argv=None):
    """
    Funkce načte parametry z příkazové řádky nebo z konfiguračního souboru.
    """

    # načítání konfigurace
    # Do argv default this way, as doing it in the functional
    # declaration sets it at compile time.
    if argv is None:
        argv = sys.argv

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file", metavar="FILE",
                             default='config')
    args, remaining_argv = conf_parser.parse_known_args()

    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults = dict(config.items("Defaults"))
    else:
        defaults = {"option": "default"}

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--option")
    args = parser.parse_args(remaining_argv)
    # v args jsou teď všechny parametry
    return args


if __name__ == "__main__":
    main()
