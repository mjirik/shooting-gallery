import cv2
import numpy as np
import matplotlib.pyplot as plt


class RedDotDetector():

    def interactive_train(self, frame):
        plt.imshow(frame)

        pts = plt.ginput(2)

        self.color_prototype_dot = frame[pts[0][::-1]]
        self.color_prototype_background = frame[pts[1][::-1]]
        self.min_area = 500

#         params = cv2.SimpleBlobDetector_Params()
#         # Change thresholds
#         params.minThreshold = int(np.mean(self.color_prototype_background))
#         params.maxThreshold = int(np.mean(self.color_prototype_dot))
#
#         # Filter by Area.
#         params.filterByArea = True
#         params.minArea = 1500
#         params.minArea = 550000
# # Create a detector with the parameters
#         ver = (cv2.__version__).split('.')
#         if int(ver[0]) < 3 :
#             self.blob_detector = cv2.SimpleBlobDetector(params)
#         else :
#             self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, frame, return_detector_image=False):

        thr = (self.color_prototype_dot.astype(np.int) +  
                self.color_prototype_background.astype(np.int)) / 2
        detector_image = (frame > thr).all(axis=2).astype(np.uint8)
        # detector_image = np.average(detector_image, axis=2)
        # detector_image = np.mean(frame, axis=2).astype(np.uint8)

        import skimage
        import skimage.morphology
        import skimage.measure
        imlab = skimage.morphology.label(detector_image, background=0)
        props = skimage.measure.regionprops(imlab)
        keypoints = []
        for prop in props:
            if prop.area > self.min_area:
                keypoints.append(KeypointFake(prop.centroid))

        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        
        
        # keypoints = self.blob_detector.detect(detector_image)


        if return_detector_image:
            return keypoints, detector_image
        return keypoints
        # return red_dot_detection(
        #     frame,
        #     return_detector_image,
        #     self.color_prototype_dot,
        #     self.color_prototype_background)


class KeypointFake():
    def __init__(self, point):
        self.pt = point


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
