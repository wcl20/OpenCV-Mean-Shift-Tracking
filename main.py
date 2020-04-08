import cv2
from tracker import ROI, meanShift_tracking

frame = None
roi = None
track_window = (0, 0, 0, 0)

# Mouse callback function
start = (0, 0)
def mouse_callback(event, x, y, flags, param):
    global frame, start, roi, track_window
    # Handle left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Reset ROI
        roi = None
        # Reset track window
        track_window = (x, y, 0, 0)
        # Define first corner
        start = (x, y)
    # Handle left click release
    if event == cv2.EVENT_LBUTTONUP:
        # Define ROI
        x, y, w, h = track_window
        roi = ROI(region=frame[y:y+h, x:x+w])
    # Handle left click drag
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        # Resize track window
        start_x, start_y = start
        track_window = (min(start_x, x), min(start_y, y), abs(start_x - x), abs(start_y - y))

def main():
    global frame, roi, track_window
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    # Bind callback function to window
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)
    while True:
        # Capture frame
        ret, frame = cap.read()
        # If there is ROI
        if not roi is None:
            # Update track window
            track_window = meanShift_tracking(frame, roi, track_window)
        # Draw rectangle
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display frame
        cv2.imshow("Video", frame)
        # Press 'q' to quit video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release video capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
