#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""
Module makes automatic callibration. You can use printed image or
digital projector.
"""

import logging
logger = logging.getLogger(__name__)
import argparse
import numpy as np
import cv2
import skimage
import skimage.measure


class Calibration():
    def __init__(self, target_file, show_function=None):
        self.target_file = target_file
        self.calibim = cv2.imread(self.target_file)          # queryImage
        self.calibim_gray = cv2.cvtColor(self.calibim, cv2.COLOR_BGR2GRAY)
        self.Minv = None
        # if show_function is None:
        #     show_function = cv2.imshow

    def get_frame(self, video_source=-1):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        res, frame = self.cap.read()
        return frame

    def black_and_white_prototype(self, frame):
        imt_black = self.calibim_gray < 10
        imt_white = self.calibim_gray > 10

        # props_b = skimage.measure.regionprops(imt_black, frame)
        # props_w = skimage.measure.regionprops(imt_white, frame)
        self.mean_black = np.mean(frame[imt_black], axis=0)
        self.mean_white = np.mean(frame[imt_white], axis=0)

    def find_surface(self, frame=None):
        Minv = None
        if frame is None:
            frame = self.get_frame()
        from matplotlib import pyplot as plt

        MIN_MATCH_COUNT = 10

        # img1 = cv2.imread('shootinggallery/box.png',0)          # queryImage
        # img2 = cv2.imread('shootinggallery/box_in_scene.png',0) # trainImage
        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img1 = self.calibim_gray



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

        self.img1 = img1
        self.img2 = img2
        self.kp1 = kp1
        self.kp2 = kp2
        self.good = good
        # dst = cv2.warpPerspective(img2, Minv, (480, 480))
        # dst = cv2.warpPerspective(image, Minv, (480, 480))
        if Minv is None:
            dst = frame
        else:
            dst = cv2.warpPerspective(frame, Minv, img1.shape)
            self.black_and_white_prototype(dst)
        # plt.imshow(dst, 'gray')  # , plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return dst

    def draw_matches(self):
        img3 = drawMatches(self.img1, self.kp1, 
                           self.img2, self.kp2, self.good)  # ,None,**draw_params)


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
