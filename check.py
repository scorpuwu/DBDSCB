import numpy as np 
# from PIL import ImageGrab
import cv2
from pynput.keyboard import Key, Controller
import time
from mss import mss

def getRedLine(frame, lower, upper):
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     mask = cv2.inRange(hsv, lower, upper)
     resized_mask = cv2.resize(mask, (300, 300))
     cv2.imshow('Red Line Mask', resized_mask) #red line viewer
     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     if contours:
          largest_contour = max(contours, key = cv2.contourArea)
          return cv2.minAreaRect(largest_contour)
     return None
     

def getWhiteBox(frame):
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
     _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

     resized_thresh = cv2.resize(thresh, (300, 300))
     cv2.imshow('White Box Mask', resized_thresh) #white box viewer
     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     #print(f"Detected {len(contours)} contours")
     if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            # print(f"Contour area: {area}")
            if 2 <= area <= 45:
                return cv2.minAreaRect(contour)
        return None
         
def isAligned(red_line, white_mark):
     red_line_center = red_line[0]
     #print("red", red_line_center)
     white_mark_center = white_mark[0]
     #print("white", white_mark_center)
     distance = np.sqrt((red_line_center[0] - white_mark_center[0]) **2 + (red_line_center[1] - white_mark_center[1])**2)

     print("distance", distance)

     if distance < 14:
          print("PRESSED SPACE")

     return distance < 14

def main():
    key = Controller()
    bbox = (890, 460, 1030, 600)
    bbox = {'top': 460, 'left': 890, 'width': 140, 'height': 140}
    lower = np.array([0, 225, 150])
    upper = np.array([0, 255, 255])

    with mss() as sct:
         while True:
               #lasttime = time.time()
               # screen = np.array(ImageGrab.grab(bbox = bbox))
               sct_img = sct.grab(bbox)
               screen = np.array(sct_img)
               frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

               red_line = getRedLine(frame, lower, upper)
               white_mark = getWhiteBox(frame)
               #print("Red Line:", red_line)  # Debugging print
               #print("White Mark:", white_mark)

               if red_line and white_mark and isAligned(red_line, white_mark):
                    key.press(Key.space)
                    key.release(Key.space)
                    time.sleep(0.1)

               resized_frame = cv2.resize(frame, (300,300))

               cv2.imshow('Screen', resized_frame)
               if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

               #print("Loop took {}".format(time.time() - lasttime))
                  
if __name__ == '__main__':
     main()

