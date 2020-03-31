import cv2

class SimpleImageProccessor:
    def __init__(self):
        pass
    def process_img(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
