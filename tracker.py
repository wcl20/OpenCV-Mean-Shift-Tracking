import cv2

# Region of Interest
class ROI:

    def __init__(self, region):
        self.region = region
        self.histogram = self.compute_histogram()

    def compute_histogram(self):
        # Convert region to HSV
        hsv = cv2.cvtColor(self.region, cv2.COLOR_BGR2HSV)
        # Compute histogram for region
        histogram = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        return histogram

def meanShift_tracking(frame, roi, track_window):
    # Blur frame
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    # Convert frame to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Compute backproject of histogram
    dst = cv2.calcBackProject([hsv],[0], roi.histogram, [0,180], 1)
    # Apply erosion (removes white noise)
    dst = cv2.erode(dst, None, iterations=2)
    # Apply dilation (counter effect of erosion)
    dst = cv2.dilate(dst, None, iterations=2)
    # Termination criteria for meanshift
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # Apply meanShift to update track window
    ret, track_window = cv2.meanShift(dst, track_window, criteria)
    return track_window
