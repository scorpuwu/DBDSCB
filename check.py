import numpy as np 
import cv2
from pynput.keyboard import Key, Controller
import time
from mss import mss
import math

def getRedLine(frame, lower, upper):
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     mask = cv2.inRange(hsv, lower, upper)
     #resized_mask = cv2.resize(mask, (300, 300))
     #cv2.imshow('Red Line Mask', resized_mask) #red line viewer
     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     if contours:
          largest_contour = max(contours, key = cv2.contourArea)
          return cv2.minAreaRect(largest_contour)
     return None
     

def getWhiteBox(frame, last_white_box, last_detection_time, memory_duration):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)
    #resized_thresh = cv2.resize(thresh, (300, 300))
    #cv2.imshow('White Box Mask', resized_thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_time = time.time()
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2 <= area <= 45:
                last_white_box = cv2.minAreaRect(contour)
                last_detection_time = current_time
                return last_white_box, last_detection_time

    # Use the last known position if less than memory_duration seconds have passed
    if current_time - last_detection_time < memory_duration:
        return last_white_box, last_detection_time

    return None, current_time

def calculate_angle(point, center):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return math.atan2(dy, dx)


def isAligned(red_line, white_mark, circle_center, angle_threshold):
    red_line_angle = calculate_angle(red_line[0], circle_center)
    white_mark_angle = calculate_angle(white_mark[0], circle_center)

    # Normalize angles to a range of 0 to 2*pi
    red_line_angle = red_line_angle % (2 * math.pi)
    white_mark_angle = white_mark_angle % (2 * math.pi)

    angle_difference = abs(red_line_angle - white_mark_angle)
    if angle_difference > math.pi:
        angle_difference = 2 * math.pi - angle_difference

    # Account for the case where angle crosses from 2*pi to 0
    if angle_difference > math.pi:
        angle_difference = 2 * math.pi - angle_difference

    #print("Angle Difference (radians):", angle_difference)

    return angle_difference < angle_threshold


def draw_debug(frame, red_line, white_mark, circle_center):
    # Draw red line
    cv2.circle(frame, (int(red_line[0][0]), int(red_line[0][1])), 5, (0, 0, 255), -1)

    # Draw white mark
    cv2.circle(frame, (int(white_mark[0][0]), int(white_mark[0][1])), 5, (255, 255, 255), -1)

    # Draw circle center
    cv2.circle(frame, circle_center, 5, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow('Debug', frame)

def main():
    key = Controller()
    bbox = {'top': 450, 'left': 880, 'width': 150, 'height': 150}
    circle_center = (bbox['left'] + bbox['width'] // 2, bbox['top'] + bbox['height'] //2)
    angle_threshold = math.radians(0.5)
    lower = np.array([0, 225, 150])
    upper = np.array([0, 255, 255])

    last_white_box = None
    last_detection_time = 0
    memory_duration = 3  # seconds

    with mss() as sct:
        while True:
            sct_img = sct.grab(bbox)
            screen = np.array(sct_img)
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

            red_line = getRedLine(frame, lower, upper)
            white_mark, last_detection_time = getWhiteBox(frame, last_white_box, last_detection_time, memory_duration)
            last_white_box = white_mark if white_mark is not None else last_white_box

            if red_line and white_mark:
                #print("Red Line Center:", red_line[0])
                #print("White Mark Center:", white_mark[0])

                if isAligned(red_line, white_mark, circle_center, angle_threshold):
                    print("Aligned - Pressing Space")
                    key.press(Key.space)
                    key.release(Key.space)
                    time.sleep(0.1)

            resized_frame = cv2.resize(frame, (300,300))
            cv2.imshow('Screen', resized_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    main()

