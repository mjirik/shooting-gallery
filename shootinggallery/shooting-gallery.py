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
from skimage.filter import threshold_otsu, gaussian_filter
import matplotlib.pyplot as plt
import pygame
import pygame.locals
import pygame.image
import yaml
import numpy as np

import blob_detection as bd
import calib


def makesurf(pixels):
    try:
        surf = pygame.surfarray.make_surface(pixels)
    except IndexError:
        (width, height, colours) = pixels.shape
        surf = pygame.display.set_mode((width, height))
        pygame.surfarray.blit_array(surf, pixels)
    return surf

class Target:

    def __init__(self, center, radius, max_score, target_file):
        self.center = np.asarray(center)
        self.radius = radius
        self.max_score = max_score
        self.image = pygame.image.load(target_file)
        self.score_coeficient = float(max_score) / float(radius)

    def get_score(self, impact_point):
        dist = np.linalg.norm(
            self.center.astype(np.float) - np.asarray(impact_point))
        score = self.max_score - (dist * self.score_coeficient)
        return max(score, 0)

    def draw_target(self, frame):
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


class FrameGetter():
    """
    unused will be removed
    """
    def __init__(self, video_source=0):
        self.video_source = video_source
        if video_source == 0:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.video_source)
            pass

    def read(self):
        res, frame = self.cap.read()
        return res, frame


class ShootingGallery():

    def __init__(self, config): # target=None, video_source=0):
        """
        Inicializační funkce. Volá se jen jednou na začátku.

        :param video_source: zdroj videa, pokud je nastaveno na číslo, je
        hledána USB kamera, je-li vložena url, předpokládá se kamera s výstupem
        do jpg.

        """
        self.config = config
        target = Target(
            config['target_center'],
            config['target_radius'],
            10,
            config['target_file']
        )
        self.calibration_surface = calib.Calibration(config['target_file'])
        video_source = config['video_source']
# create video capture
        # video_source = 0
        # video_source = "http://192.168.1.60/snapshot.jpg"
        # self.cap = cv2.VideoCapture(video_source)
        self.cap = FrameGetter(video_source)
        # if target is None:
        #     self.target = Target([300, 300], 200, 10)
        # else:
        #     self.target = target
        self.status_text = ""
        self.target = target
        # self.default_mode = 'paper'
        self.mode = 'projector' 
        self.mode = 'paper' 

    def __show_keypoints(self, keypoints, frame):
        for i, keypoint in enumerate(keypoints):
            cx = int(keypoint.pt[0])
            cy = int(keypoint.pt[1])
# each next point is bigger, just to recognize them
            cv2.circle(frame, (cx, cy), 10 + i,
                       (100, 255, 255),
                       -1)
            # cv2.circle(frame,keypoints[0].pt,5,255,-1)
            # print self.target.get_score([cx,cy])

            if i == 0:
                self.status_text = "%.2f" % (
                    self.target.get_score([cx, cy]))
                print self.status_text
        return frame

    def tick(self):
        """
        Tato funkce se vykonává opakovaně
        """
        # read the frames
        ret, frame = self.cap.read()
        if ret:

            wframe = cv2.warpPerspective(
                    frame, 
                    self.calibration_surface.Minv, 
                    tuple(self.config['resolution']))

            # keypoints = bd.red_dot_detection(frame)
            # keypoints = bd.diff_dot_detection(frame, self.init_frame)
            # keypoints, frame = self.dot_detector.detect(frame, True)
            keypoints = self.dot_detector.detect(wframe)

            if self.mode == 'paper':
                # smooth it
                wframe = self.__show_keypoints(keypoints, wframe)
                # Show it, if key pressed is 'Esc', exit the loop

                self.print_status(wframe)
                self.target.draw_target(wframe)
                # cv2.imshow('frame', frame)
                wframe = np.transpose(wframe, axes=[1, 0, 2])
                surf = makesurf(wframe)
            elif self.mode == 'projector':
# TODO projector mode
                wframe = self.__show_keypoints(keypoints, self.target.image)
                # Show it, if key pressed is 'Esc', exit the loop

                self.print_status(wframe)
                self.target.draw_target(wframe)
                # cv2.imshow('frame', frame)
                wframe = np.transpose(wframe, axes=[1, 0, 2])
                surf = makesurf(wframe)
            self.screen.blit(surf, (0,0))
            pygame.display.flip()        
            self.clock.tick(10)                                  # omezení maximálního počtu snímků za sekundu
        return True

    def calibration(self):
        # get transformation
        self.__calib_show_function(self.calibration_surface.calibim)
        _, frame = self.cap.read()
        self.init_frame = self.calibration_surface.find_surface(frame)
        # get image with red point
        self.clock.tick(500)                                  # omezení maximálního počtu snímků za sekundu
        _, frame = self.cap.read()


        frame_with_dot = cv2.warpPerspective(
                frame,
                self.calibration_surface.Minv, 
                tuple(self.config['resolution'])
                # (480, 480)
                # self.calibration_surface.calibim_gray.shape
                )# (480, 480))
        plt.imshow(frame_with_dot)
        print "Klikněte na bod laseru a pak kamkoliv do ostatní plochy"
        
        self.dot_detector = bd.RedDotDetector()
        self.dot_detector.interactive_train(frame_with_dot) #pts[0], pts[1])

    def __calib_show_function(self, frame):

        frame = np.transpose(frame, axes=[1, 0, 2])
        surf = makesurf(frame)
        self.screen.blit(surf, (0,0))
        pygame.display.flip()        
        self.clock.tick(500)                                  # omezení maximálního počtu snímků za sekundu


    def run(self):
        """
        funkce opakovaně volá funkci tick
        """
        pygame.init()
        self.screen = pygame.display.set_mode(self.config['resolution'])         # vytvoření okna s nastavením jeho velikosti
        self.projector = pygame.display.set_mode(self.config['resolution'])         # vytvoření okna s nastavením jeho velikosti

        pygame.display.set_caption("Shooting Gallery")               # nastavení titulku okna
        self.background = pygame.Surface(self.screen.get_size())      # vytvoření vrstvy pozadí
        self.background = self.background.convert()                   # převod vrstvy do vhodného formátu
        self.background.fill((0,0,255))                 
        self.clock = pygame.time.Clock()                         # časování

        pygame.display.flip()        
        self.clock.tick(5)                                  # omezení maximálního počtu snímků za sekundu
        self.calibration()


        print('Run')
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
            (10, 100), font, 4, (100, 100, 255), 4)  # ,2,cv2.LINE_AA)


def main():
    config = get_params()
    print config
    # convert to ints
    sh = ShootingGallery(config)
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
    args = conf_parser.parse_args()

    # if args.conf_file:
    stream = open(args.conf_file, 'r')
    config = yaml.load(stream)

        # config = ConfigParser.SafeConfigParser()
        # config.read([args.conf_file])
        # defaults = dict(config.items("Defaults"))
    # else:
    #     config = {"option": "default"}

    # # Parse rest of arguments
    # # Don't suppress add_help here so it will handle -h
    # parser = argparse.ArgumentParser(
    #     # Inherit options from config_parser
    #     parents=[conf_parser]
    # )
    # parser.set_defaults(**defaults)
    # parser.add_argument("--option")
    # args = parser.parse_args(remaining_argv)
    # # v args jsou teď všechny parametry
    return config 


if __name__ == "__main__":
    main()
