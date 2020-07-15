import cv2

## video parameters
video_size = 1000
# fps = 24    # 24 frames per second, for Lydia's .wmv files
fps = 40    # NN: raspi camera default frame rate = 30 fps

# parameters to detect rat presence
min_area = 2000 # 200, minimum size (in pixels) for a region of an image to be considered a rat, arbitrary
max_area = 3000
threshold = 58 # 40-60, threshold for fixed-level thresholding, value is critical if it is too high, get drop-outs
ksize = 21   # Gaussian blur parameter, higher values wash out the tail resulting in a better position for the centroid


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def findCountours(frame, backgroundFrame):
# creates a thresholded image and finds any contours, i.e. the 'rat'
# from http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

    # resize the frame, convert it to grayscale, and blur it
    # frame = resize(frame, width = video_size) # resize image to setting
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    
    # compute the absolute difference between the current frame and background frame
    frameDelta = cv2.absdiff(backgroundFrame, gray)
    thresh = cv2.threshold(frameDelta, threshold, 255, cv2.THRESH_BINARY)[1]
    
# uncomment to examine effect of changing parameters
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.waitKey(1) & 0xFF

    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (contours, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return frame, contours

def addBoxes(boxedFrame, boxes):   
# adds boxes to the frame image 
    for box in boxes:
        cv2.rectangle(boxedFrame, box[0], box[1], boxGray1, 1) 
    return boxedFrame

# resizes image, from  Adrian Rosebrock's imutils package
def resize(image, width=None, height=None, inter=cv2.INTER_AREA): 
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    # (h, w) = image.shape[:2]

    w = image.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
    h = image.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
