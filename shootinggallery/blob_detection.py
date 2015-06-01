#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage


class RedDotDetector():
    """
    Lze nastavit interaktivně, nebo jen ručně nastavit práh, 
    případně min_area_coeficient
    """

    def __init__(self):
        self.detection_mode = 'continuous'
        self.detection_mode = 'falling'
        self.detection_mode = 'rising'
        self.prev_keypoints = []
        self.min_area_coeficient = 0.5
        self.gaussian_filter_sigma = 0.9

    def interactive_train(self, frame, min_area_coeficient=None):
        """
        User click on blob and on the background and all is trained 
        """
        if min_area_coeficient is not None:
            self.min_area_coeficient = min_area_coeficient
        plt.imshow(frame)

        pts = plt.ginput(2)

        plt.close()
        self.color_prototype_dot = frame[pts[0][::-1]]
        self.color_prototype_background = frame[pts[1][::-1]]

        self.thr = (self.color_prototype_dot.astype(np.int) +  
                self.color_prototype_background.astype(np.int)) / 2
        self.train_min_area(frame, pts)


    def train_min_area(self, frame, pts=None):
        """
        estimate minimal area

        """
        detector_image = (frame > self.thr).all(axis=2).astype(np.uint8)
        imlabel = skimage.morphology.label(detector_image)

        lab = imlabel[pts[0][::-1]]
        sm = np.sum(imlabel==lab)

        self.min_area = int(self.min_area_coeficient * sm)

    def detect(self, frame, return_debug_image=False):
        # from skimage.filter import gaussian_filter
        if self.gaussian_filter_sigma is not None:
            frame = cv2.GaussianBlur(frame,(5,5),self.gaussian_filter_sigma)
            # frame = gaussian_filter(frame, self.gaussian_filter_sigma)

        thr = self.thr
        detector_image = (frame > thr).all(axis=2).astype(np.uint8)
        # detector_image = np.average(detector_image, axis=2)
        # detector_image = np.mean(frame, axis=2).astype(np.uint8)

        import skimage
        import skimage.morphology
        import skimage.measure
        imlab = skimage.morphology.label(detector_image, background=0)
        props = skimage.measure.regionprops(imlab + 1)
        print np.max(imlab), np.min(imlab), len(props)
        keypoints = []
        for prop in props:
            if prop.area > self.min_area:
                keypoints.append(KeypointFake(prop.centroid))

        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        
        
        # keypoints = self.blob_detector.detect(detector_image)
        okeypoints = keypoints

        if self.detection_mode is 'rising':
            if len(self.prev_keypoints) == 0:
                okeypoints = keypoints
            else:
                okeypoints = []
        elif self.detection_mode is 'falling':
            if len(keypoints) == 0:
                okeypoints = self.prev_keypoints
            else:
                okeypoints = []

        self.prev_keypoints = keypoints

        if return_debug_image:
            return okeypoints, detector_image, imlab
        return okeypoints
        # return red_dot_detection(
        #     frame,
        #     return_detector_image,
        #     self.color_prototype_dot,
        #     self.color_prototype_background)


class KeypointFake():
    def __init__(self, point):
        self.pt = point[::-1]


def diff_dot_diff_detection(frame, init_frame):
    frame = frame - init_frame

    frame = cv2.blur(frame, (5, 5))
    blob_detector = cv2.FeatureDetector_create("SimpleBlob")
    detector_image = np.minimum(
        color_filter(frame, [255, 255, 255]),
        color_filter(frame, [10, 10, 255]))

    keypoints = blob_detector.detect(detector_image)
    return keypoints


def red_dot_detection(
        frame,
        return_detector_image=False,
        color_prototype_dot=[255, 255, 255],
        color_prototype_background=[10, 10, 255]):

    frame = cv2.blur(frame, (5, 5))

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # thresh = cv2.inRange(hsv,np.array((0, 80, 80)), np.array((20, 255, 255)))
    # thresh2 = thresh.copy()

    blob_detector = cv2.FeatureDetector_create("SimpleBlob")
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # inv_r_channel = 255 - frame[:,:,2]
    # red or white colors
    # detector_image = np.minimum(
    #     color_filter(frame, color_prototype_dot),
    #     color_filter(frame, color_prototype_background))
    thr = (color_prototype_dot.astype(np.int) +  
            color_prototype_background.astype(np.int)) / 2
    # simil_dot = color_filter(frame, color_prototype_dot)
    # simil_background = color_filter(frame, color_prototype_background)
    # detector_image = (frame > thr).astype(np.uint8)
    detector_image = (frame>thr).all(axis=2).astype(np.uint8)
    # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
    # detector_image = (
    #         color_filter(frame, color_prototype_dot) )
    # detector_image = (
    #         color_filter(frame, color_prototype_dot).astype(np.int) -
    #         color_filter(frame, color_prototype_background).astype(np.int)).astype(np.int)

    # detector_image = np.average(detector_image, axis=2)
    import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
    
    keypoints = blob_detector.detect(detector_image)
    # convert to hsv and find range of colors
    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # thresh=cv2.inRange(hsv,np.array((230, 80, 80)), np.array((255, 255, 255)))
    # thresh2 = thresh.copy()

    if return_detector_image:
        return keypoints, detector_image

    return keypoints


def color_filter(image, color_prototype):
    suma = np.zeros(image.shape[:2])
    for i, channel in enumerate(color_prototype):
        diff = (image[:, :, i].astype(np.float) - channel).astype(np.float)
        suma += diff ** 2
        # print 'ch ', channel, ' ', np.max(image[:,:,i]), ' ', np.max(diff)

    retval_fl = (suma ** 0.5) / 2.0  # 2*(1/len(color_prototype)) #1.43
    retval = retval_fl.astype(image.dtype)

    # cb, cg, cr = image[0,0,:]
    # print "rgb %i %i %i sum %i sqrts %s" % (
    #     cb, cg, cr,
    # image[0,0,0],
    # image[0,0,1],
    # image[0,0,2],
    #     suma[0,0],
    #     str(retval_fl[0,0]))
    #
    # dst = float(cb)**2 + float(cg)**2 + (float(cr) - **2
    # print
    return retval


def main():
    # create video capture
    cap = cv2.VideoCapture(0)

    while(1):

        # read the frames
        _, frame = cap.read()

        keypoints, detector_image = red_dot_detection(
            frame, return_detector_image=True)

        # smooth it
        for keypoint in keypoints:
            cx = int(keypoint.pt[0])
            cy = int(keypoint.pt[1])
            cv2.circle(frame, (cx, cy), 5, 255, -1)
            # cv2.circle(frame,keypoints[0].pt,5,255,-1)

        # Show it, if key pressed is 'Esc', exit the loop
        cv2.imshow('frame', frame)
        cv2.imshow('thresh', detector_image)
        if cv2.waitKey(33) == 27:
            break

# Clean up everything before leaving
    cv2.destroyAllWindows()
    cap.release()
if __name__ == "__main__":
    main()
