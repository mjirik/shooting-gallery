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

import blob_detection as bd


class Target:

    def __init__(self, center, radius, max_score, target_file):
        self.center = np.asarray(center)
        self.radius = radius
        self.max_score = max_score
        self.target_file = target_file
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

    def find_target(self, image):
        import cv2
        from matplotlib import pyplot as plt

        MIN_MATCH_COUNT = 10

        # img1 = cv2.imread('shootinggallery/box.png',0)          # queryImage
        # img2 = cv2.imread('shootinggallery/box_in_scene.png',0) # trainImage
        img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img1 = cv2.imread(self.target_file)          # queryImage
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


# Initiate SIFT detector
        sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img3 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.CV_AA)

            Minv, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            self.Minv = Minv
        else:
            print "Not enough matches are found - %d/%d" % (
                len(good), MIN_MATCH_COUNT)
            matchesMask = None

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor = None,
            matchesMask=matchesMask,  # draw only inliers
            flags = 2)

        img3 = drawMatches(img1, kp1, img2, kp2, good)  # ,None,**draw_params)
        # dst = cv2.warpPerspective(img2, Minv, (480, 480))
        dst = cv2.warpPerspective(image, Minv, (480, 480))
        plt.imshow(dst, 'gray')  # , plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return dst


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)), (int(x2)+cols1, int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class FrameGetter():
    """
    unused will be removed
    """
    def __init__(self, video_source=0):
        self.video_source = video_source
        if video_source == 0:
            self.cap = cv2.VideoCapture(0)
        else:
            pass

    def read(self):
        if self.video_source == 0:
            res, frame = self.cap.read()
        else:
            self.cap = cv2.VideoCapture(self.video_source)
            res, frame = self.cap.read()
        return res, frame


class ShootingGallery():

    def __init__(self, target=None, video_source=0):
        """
        Inicializační funkce. Volá se jen jednou na začátku.

        :param video_source: zdroj videa, pokud je nastaveno na číslo, je
        hledána USB kamera, je-li vložena url, předpokládá se kamera s výstupem
        do jpg.

        """
# create video capture
        # video_source = 0
        # video_source = "http://192.168.1.60/snapshot.jpg"
        # self.cap = cv2.VideoCapture(video_source)
        self.cap = FrameGetter(video_source)
        if target is None:
            self.target = Target([300, 300], 200, 10)
        else:
            self.target = target
        self.status_text = ""
        pass

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
        _, frame = self.cap.read()
        frame = cv2.warpPerspective(frame, self.target.Minv, (480, 480))

        # keypoints = bd.red_dot_detection(frame)
        # keypoints = bd.diff_dot_detection(frame, self.init_frame)
        keypoints, frame = self.dot_detector.detect(frame, True)

        # smooth it
        frame = self.__show_keypoints(keypoints, frame)
        # Show it, if key pressed is 'Esc', exit the loop

        self.print_status(frame)
        self.target.draw_target(frame)
        cv2.imshow('frame', frame)
        # cv2.imshow('thresh',inv_r_channel)
        if cv2.waitKey(33) == 27:
            return False
        return True

    def calibration(self):
        # get transformation
        _, frame = self.cap.read()
        self.init_frame = self.target.find_target(frame)
        # get image with red point
        _, frame = self.cap.read()
        frame_with_dot = cv2.warpPerspective(frame, self.target.Minv, (480, 480))
        plt.imshow(frame_with_dot)
        print "Klikněte na bod laseru a pak kamkoliv do ostatní plochy"
        
        self.dot_detector = bd.RedDotDetector()
        self.dot_detector.interactive_train(frame_with_dot) #pts[0], pts[1])

    def run(self):
        """
        funkce opakovaně volá funkci tick
        """
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
    args = get_params()
    print vars(args)
    # convert to ints
    print json.loads(args.target_center)
    target = Target(
        json.loads(args.target_center),
        json.loads(args.target_radius),
        10,
        args.target_file
    )
    sh = ShootingGallery(target, video_source=json.loads(args.video_source))
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
