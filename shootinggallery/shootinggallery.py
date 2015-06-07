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
import random
import os

import blob_detection as bd
import calib
import expocomp
import targets
from targets import Target 
from cameraio import FrameGetter, np2surf

_sound_library = {}


def normrnd(val ,scale):
    val = np.asarray(val)
    scale = np.asarray(scale)
    zrs =  scale == 0
    scale[zrs] = 1
    rnd = np.random.normal(val, scale)
    rnd [zrs] = val[zrs]
    return rnd

def play_sound(path):
  global _sound_library
  sound = _sound_library.get(path)
  if sound == None:
    canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
    sound = pygame.mixer.Sound(canonicalized_path)
    _sound_library[path] = sound
  sound.play()

def read_surf(infoin):

    if infoin is None or infoin == 'None':
        return None, None

    info = {'invert_intensity' : False , 'offset': [0, 0], 'intensity_multiplier': None}
    info.update(infoin)
    
    surface = pygame.image.load(info['impath'])
    # if self.config['flip']:
    #     surface = pygame.transform.flip(surface, True, False)
    # pygame.transform.scale(surface)
    intensity_multiplier = info['intensity_multiplier']
    surface = pygame.transform.rotozoom(surface, 0, info['zoom'])
    if info['invert_intensity']:
        print 'invert'
        surface = targets.inverted(surface)
    if intensity_multiplier is not None:
        print 'multi'
        ndim = pygame.surfarray.pixels3d(surface) 
        for (x,y,z), value in np.ndenumerate(ndim):
            ndim[x,y,z] = value * intensity_multiplier
    return surface, info['offset']

class GameModel():
    def __init__(self, game_time=60, free=False, nshoots=10):
        self.state = 'waiting'
        if free:
            self.state = 'free'
        self.score = 0
        self.prev_score = 0
        self.time = 0
        self.game_time = game_time * 1000
        self.scoreboard_time = 10 * 1000
        self.nshoots_max = nshoots
        self.nshoots = nshoots

    def start(self):
        if self.state == 'waiting':
            self.state = 'running'
            self.time = self.game_time
            self.score = 0
            self.prev_score = 0
            self.nshoots = self.nshoots_max

    def add_score(self, score):
        score = int(score)
        if self.state in ('running', 'free'):
            self.score += score
            self.prev_score = score
            self.nshoots -= 1
        

    def update(self, deltat):
        if self.state == 'running':
            self.time -= deltat
            if self.time < 0 or self.nshoots <=0:
                self.state = 'scoreboard'
                self.time = self.scoreboard_time

        elif self.state == 'scoreboard':
            self.time -= deltat
            if self.time < 0:
                self.state = 'waiting'
                self.time = 0 
                self.nshoots = self.nshoots_max

    def get_status_text(self):
        if self.state == 'free':
            # status_text = "%.2" % (self.prev_score)
            status_text = "%i" % (int(self.prev_score))
        elif self.state == 'scoreboard':
            status_text = "Score: %i" %(int(self.score))
        else:
            status_text = "%3d %2d %3d %2d" %(
                    int(self.score), 
                    int(self.prev_score),
                    int(self.time / 1000),
                    int(self.nshoots),
                    )
        return status_text




class ShootingGallery():

    def __init__(self, config): # target=None, video_source=0):
        """
        Inicializační funkce. Volá se jen jednou na začátku.

        :param video_source: zdroj videa, pokud je nastaveno na číslo, je
        hledána USB kamera, je-li vložena url, předpokládá se kamera s výstupem
        do jpg.


        Konfigurace cile je dana v kofiguračním souboru v poli targets.
        Tyto hodnoty pak mohou být řízeny ještě v každé jednotlivé scéně. 
        pomocí scenes - targets - id - config

        """
        self.config = config
        # target = Target(
        #     config['target_center'],
        #     config['target_radius'],
        #     10,
        #     config['target_file']
        # )
        # target = Target(**config['targets']['paper_target'])
        # targets = Targets()
        targets = pygame.sprite.Group()
        # targets.add(target)
        self.calibration_surface = calib.Calibration(config['calibration_image'])
        video_source = config['video_source']
# create video capture
        # self.cap = FrameGetter(video_source, resolution=[800, 600])
        self.cap = FrameGetter(
                video_source, 
                resolution=config['video_source_resolution'])
        self.elapsed = 0
        self.game = GameModel()
        self.status_text = ""
        self.targets = targets
        # self.default_mode = 'paper'
        self.mode = 0
        self.debugmode = 'N'

    def __process_keypoints(self, keypoints, screen):

        # print keypoints
        # if len(keypoints) > 0:
        #     pass
        for i, keypoint in enumerate(keypoints):
            cx = int(keypoint.pt[0])
            cy = int(keypoint.pt[1])
# each next point is bigger, just to recognize them
            pygame.draw.circle(screen, (100, 255, 255), (cx, cy), i+10, 5)

            # cv2.circle(frame, (cx, cy), 10 + i,
            #            (100, 255, 255),
            #            -1)

            if i == 0:
                print 'booom'
                play_sound('shootinggallery/sound/Gun_Shot-Marvin-1140816320.wav')
                self.game.start()
                sc = 0.0
                for tg in self.targets:
                    sc += tg.get_score([cx, cy])
                self.game.add_score(sc)
                # self.status_text = "%.2f" % (sc)
                # print self.status_text
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

    def __blit_surf_or_frame(self, surf, frame):
        pos = (0, 0)
        if surf == 'frame':
            surf = np2surf(frame)

        if surf is not None:
            self.screen.blit(surf, pos)

    def __generate_target(self, deltat):
        if self.elapsed is not None:
            self.elapsed -= deltat
            if self.elapsed < 0:
                scene_config = self.config['scenes'][self.mode]
                # which target is used
                rng = random.randint(0,len(scene_config['targets'])-1)

            
                new_tg_key = scene_config['targets'][rng]['target_key']
                new_tg_gen_config = {'mean_time': 10, 'time_var':4}
                new_tg_gen_config.update(scene_config['targets'][rng])
                tg_config = self.config['targets'][new_tg_key]
                if "config" in  scene_config['targets'][rng].keys():
                    tg_config.update(scene_config['targets'][rng]['config'])
                if new_tg_gen_config['mean_time'] == 'None':
                    self.elapsed = None
                else:
                    # self.elapsed = (0.5 + random.random()) * 1000 * new_tg_gen_config['mean_time']
                    self.elapsed = np.random.normal(new_tg_gen_config['mean_time'], new_tg_gen_config['time_var']) * 1000
                
                self.targets.add(Target(**tg_config))

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

            keypoints, det_img, lab_img = self.dot_detector.detect(
                    wframe,
                    return_debug_image=True
                    )

            if self.debugmode == "N":
                self.__generate_target(deltat)
                self.__blit_surf_or_frame(self.background, wframe)
                # self.screen.blit(makesurf(frame), (0, 0))
                self.targets.update(deltat)
                self.targets.draw(self.screen)
                self.__process_keypoints(keypoints, self.screen)
                self.game.update(deltat)
                self.status_text = self.game.get_status_text()
                self.print_status(self.screen)

            # print self.debugmode
            if self.debugmode == "D":
                self.screen.blit(np2surf(frame), (0, 0))
            elif self.debugmode == "F":
                self.screen.blit(np2surf(cframe), (0, 0))
            elif self.debugmode == "G":
                self.screen.blit(np2surf(wframe), (0, 0))
            elif self.debugmode == "H":
                self.screen.blit(np2surf(det_img*100), (0, 0))
            elif self.debugmode == "J":
                self.screen.blit(np2surf((lab_img + 1)*40), (0, 0))
            pygame.display.flip()        
        return True

    def __prepare_scene(self, i):
        self.mode = i
        scene_config = {
                'fontsize': 110,
                'free_game': False,
                'game_time': 60, 
                'nshoots': 5}
        scene_config.update(self.config['scenes'][i])

        self.background, self.background_offset = read_surf(scene_config['background'])
        self.foreground, self.foreground_offset = read_surf(scene_config['foreground'])
        self.fontsize = scene_config['fontsize']
        self.targets.empty()
        self.elapsed = 0
        self.game = GameModel(free=scene_config['free_game'], 
                              game_time=scene_config['game_time'], 
                              nshoots=scene_config['nshoots'])

        # read_surf(


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
                        
                elif event.key == pygame.locals.K_b:
                    print 'calibration'
                    self.calib_blob()
                    # self.__prepare_scene(5)

    def calib_blob(self):
        _, frame = self.cap.read()
        cframe, wframe = self.__camera_image_processing(frame)
        self.dot_detector.interactive_train(wframe, min_area_coeficient=0.6) #pts[0], pts[1])

    def calibration(self, interactive=True):
        # get transformation
# show calibration image (for projector mode)
        self.__calib_show_function(self.calibration_surface.calibim)
        self.clock.tick(500)                                  # omezení maximálního počtu snímků za sekundu
        _, frame = self.cap.read()
        self.aec = expocomp.AutomaticExposureCompensation()
        
        self.aec.set_ref_image(frame)
        self.aec.set_area(20, 20)
        self.init_frame = self.calibration_surface.find_surface(frame)
        # get image with red point
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

        if interactive:
            self.dot_detector.interactive_train(frame_with_dot, min_area_coeficient=0.4) #pts[0], pts[1])
        else:
            # self.dot_detector.min_area_coeficient = 0.4
            # self.dot_detector.thr = self.calibration_surface.max_white
            self.dot_detector.thr = self.calibration_surface.mean_white + \
                    self.config['auto_target_calibration_white_var_alpha'] *\
                    (self.calibration_surface.var_white**0.5)
            self.dot_detector.min_area = \
                    self.config['auto_target_calibration_min_area']

    def __calib_show_function(self, frame):

        surf = np2surf(frame)
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
        self.calibration(interactive=False)
        self.__prepare_scene(self.mode)


        print('Run')
        while self.keepGoing:

            self.tick()

        self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.cap.release()

    def print_status(self, screen):
        # self.status_text = "S " + self.status_text 
        font=pygame.font.Font(None, self.fontsize)
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
                             default='config.yml')
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
