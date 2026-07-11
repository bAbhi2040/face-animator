from PIL import Image
import os
import cv2

file_path = input("Copy and paste the image's file path: ").strip('"')
if not os.path.exists(file_path):
    print('An error occured')
else:
    read_image = cv2.imread(file_path)

    if read_image is None:
        print('OpenCV image opening error')
    else:
        cv2.imshow('image', read_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


