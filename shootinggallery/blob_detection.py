import cv2
import numpy as np


def red_dot_detection(frame, return_detector_image=False):

    frame = cv2.blur(frame, (5, 5))

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # thresh = cv2.inRange(hsv,np.array((0, 80, 80)), np.array((20, 255, 255)))
    # thresh2 = thresh.copy()

    blob_detector = cv2.FeatureDetector_create("SimpleBlob")
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # inv_r_channel = 255 - frame[:,:,2]
    # red or white colors
    detector_image = np.minimum(
        color_filter(frame, [255, 255, 255]),
        color_filter(frame, [10, 10, 255]))
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
