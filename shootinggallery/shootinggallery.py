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
import expocomp
from targets import Target, Targets


def makesurf(pixels):
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
        surf = pygame.display.set_mode((width, height))
        pygame.surfarray.blit_array(surf, pixels)
    return surf





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

def read_surf(self, info):

    if info is None or info == 'None':
        return None, None
    
    surface = pygame.image.load(info['impath'])
    # if self.config['flip']:
    #     surface = pygame.transform.flip(surface, True, False)
    # pygame.transform.scale(surface)
    surface = pygame.transform.rotozoom(surface, 0, info['zoom'])
    return surface, info['offset']

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
        target = Target(**config['targets']['paper_target'])
        # targets = Targets()
        targets = pygame.sprite.Group()
        targets.add(target)
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
        self.targets = targets
        # self.default_mode = 'paper'
        self.mode = 'projector' 
        self.mode = 'paper' 
        self.mode = 1
        self.debugmode = 'N'

    def __show_keypoints(self, keypoints, screen):
        for i, keypoint in enumerate(keypoints):
            cx = int(keypoint.pt[0])
            cy = int(keypoint.pt[1])
# each next point is bigger, just to recognize them
            pygame.draw.circle(screen, (100, 255, 255), (cx, cy), i+10, 5)

            # cv2.circle(frame, (cx, cy), 10 + i,
            #            (100, 255, 255),
            #            -1)

            if i == 0:
                sc = 0.0
                for tg in self.targets:
                    sc += tg.get_score([cx, cy])
                self.status_text = "%.2f" % (sc)
                print self.status_text
        return screen

    def __camera_image_processing(self, frame):
        """
        :cframe: compensated frame
        :wframe: warped frame
        """

        cframe = self.aec.compensate(frame)

        if self.calibration_surface.Minv is None:
            wframe = cframe
        else:
            wframe = cv2.warpPerspective(
                    cframe, 
                    self.calibration_surface.Minv, 
                    tuple(self.config['resolution']))
        sh = self.calibration_surface.calibim_gray.shape
        wframe[sh[0]:, :] = 0
        wframe[:, sh[1]:] = 0
        return wframe, cframe



    def tick(self):
        """
        Tato funkce se vykonává opakovaně
        """
        # read the frames
        ret, frame = self.cap.read()
        if ret:
            deltat = self.clock.tick(25)                                  # omezení maximálního počtu snímků za sekundu

            wframe, cframe = self.__camera_image_processing(frame)
            self.event_processing()


            # keypoints = bd.red_dot_detection(frame)
            # keypoints = bd.diff_dot_detection(frame, self.init_frame)
            # keypoints, frame = self.dot_detector.detect(frame, True)
            keypoints, det_img, lab_img = self.dot_detector.detect(
                    wframe,
                    return_debug_image=True
                    )

            if self.debugmode == "N":
                if self.mode == 1:
                    # smooth it
                    # Show it, if key pressed is 'Esc', exit the loop

                    # self.print_status(wframe)
                    # self.targets.draw(wframe)
                    # cv2.imshow('frame', frame)
                    # wframe = np.transpose(wframe, axes=[1, 0, 2])
                    # surf = makesurf(wframe)
                    self.targets.update(deltat)
                    self.targets.draw(self.screen)
                    self.__show_keypoints(keypoints, self.screen)
                    self.print_status(self.screen)
                elif self.mode == 'projector':
# TODO projector mode
                    wframe = self.__show_keypoints(keypoints, self.target.image)
                    # Show it, if key pressed is 'Esc', exit the loop

                    self.print_status(wframe)
                    self.target.draw(wframe)
                    # cv2.imshow('frame', frame)
                    wframe = np.transpose(wframe, axes=[1, 0, 2])
                    surf = makesurf(wframe)
                # print self.mode
                # self.screen.blit(surf, (0,0))

            if self.debugmode == "D":
                self.screen.blit(makesurf(frame), (0, 0))
            elif self.debugmode == "F":
                self.screen.blit(makesurf(cframe), (0, 0))
            elif self.debugmode == "G":
                self.screen.blit(makesurf(wframe), (0, 0))
            elif self.debugmode == "H":
                self.screen.blit(makesurf(det_img*100), (0, 0))
            elif self.debugmode == "J":
                self.screen.blit(makesurf((lab_img + 1)*40), (0, 0))
            pygame.display.flip()        
        return True

    def __prepare_scene(self, i):
        self.mode = i
        import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        scene_config = self.config['scenes'][0]
        # read_surf(

        pass

    def event_processing(self):
        for event in pygame.event.get():
            # any other key event input
            if event.type == pygame.locals.QUIT:
                done = True        
            elif event.type == pygame.locals.KEYDOWN:
                if event.key == pygame.locals.K_ESCAPE:
                    self.keepGoing = False
                elif event.key == pygame.locals.K_1:
                    print "hi world mode"


                # if event.key == pygame.K_ESCAPE:
                #     self.keepGoing = False                       # ukončení hlavní smyčky
                elif event.key == pygame.locals.K_SPACE:
                    self.snapshot()
                elif event.key == pygame.locals.K_KP0:
                    self.__prepare_scene(0)
                elif event.key == pygame.locals.K_KP1:
                    self.__prepare_scene(1)
                elif event.key == pygame.locals.K_KP2:
                    self.__prepare_scene(2)
                elif event.key == pygame.locals.K_KP3:
                    self.__prepare_scene(3)
                elif event.key == pygame.locals.K_KP4:
                    self.__prepare_scene(4)
                elif event.key == pygame.locals.K_KP5:
                    self.__prepare_scene(5)
                elif event.key == pygame.locals.K_KP6:
                    self.__prepare_scene(6)
                elif event.key == pygame.locals.K_KP7:
                    self.__prepare_scene(7)
                elif event.key == pygame.locals.K_KP8:
                    self.__prepare_scene(8)
                elif event.key == pygame.locals.K_KP9:
                    self.__prepare_scene(9)
                elif event.key == pygame.locals.K_i:
                    print self.cap.get(cv.CV_CAP_PROP_MODE)
                    print self.cap.get(cv.CV_CAP_PROP_BRIGHTNESS)
                    print self.cap.get(cv.CV_CAP_PROP_CONTRAST)
                    print self.cap.get(cv.CV_CAP_PROP_SATURATION)
                    print self.cap.get(cv.CV_CAP_PROP_GAIN)
                    import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

                elif event.key == pygame.locals.K_d:
                    self.debugmode = 'D' 
                    print "debugmode D"
                elif event.key == pygame.locals.K_f:
                    self.debugmode = 'F' 
                    print "debugmode F"
                elif event.key == pygame.locals.K_g:
                    self.debugmode = 'G' 
                    print "debugmode G"
                elif event.key == pygame.locals.K_h:
                    self.debugmode = 'H' 
                    print "debugmode H"
                elif event.key == pygame.locals.K_j:
                    self.debugmode = 'J' 
                    print "debugmode J"
                elif event.key == pygame.locals.K_k:
                    self.debugmode = 'K' 
                    print "debugmode K"
                elif event.key == pygame.locals.K_n:
                    self.debugmode = 'N' 
                    print "debugmode N"
                elif event.key == pygame.locals.K_c:
                    print 'calibration'
                    self.calibration()
                        
                    # self.__prepare_scene(5)

    def calibration(self):
        # get transformation
# show calibration image (for projector mode)
        self.__calib_show_function(self.calibration_surface.calibim)
        _, frame = self.cap.read()
        self.aec = expocomp.AutomaticExposureCompensation()
        
        self.aec.set_ref_image(frame)
        self.aec.set_area(20, 20)
        self.init_frame = self.calibration_surface.find_surface(frame)
        # get image with red point
        self.clock.tick(500)                                  # omezení maximálního počtu snímků za sekundu
        _, frame = self.cap.read()

        if self.calibration_surface.Minv is None:
            frame_with_dot = frame
            print("Calibration failed")
        else:
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
        self.keepGoing = True

        pygame.display.flip()        
        self.clock.tick(5)                                  # omezení maximálního počtu snímků za sekundu
        self.calibration()


        print('Run')
        while self.keepGoing:
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

    def print_status(self, screen):
        self.status_text = "TEST"
        font=pygame.font.Font(None,110)
        scoretext=font.render(self.status_text, 3,(50,150,50))
        screen.blit(scoretext, (10, 10))
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(
        #     frame,
        #     self.status_text,
        #     (10, 100), font, 4, (100, 100, 255), 4)  # ,2,cv2.LINE_AA)


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
