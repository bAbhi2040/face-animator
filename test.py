from PIL import Image
import os
import cv2
import numpy as np

i = 1

file_path = input("Copy and paste the image's file path: ").strip('"')
if not os.path.exists(file_path):
    print('An error occured')
else:
    src = cv2.imread(file_path)
    output = src.copy()

    if src is None:
        print('OpenCV image opening error')
    else:
        userChoice = input('Default, grayscale, blurred, or with Hough circles? ').lower()

        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        def blur_generator(i, maxKernalLength, image):
            while i < maxKernalLength:
                i += 2
                blurred_image = cv2.blur(image, ksize=(i, i))
            return blurred_image
        
        if userChoice == 'default':
            cv2.imshow('Default image', src)
        elif userChoice == 'grayscale':
            cv2.imshow('Grayscaled image', gray)
        elif userChoice == 'blurred':
            maxKernalLength = int(input('What is the max kernal length (must be odd)?: '))
            blurred_image = blur_generator(i, maxKernalLength, src)
            cv2.imshow('Blurred image', blurred_image)
        elif userChoice == 'hough circles':
            gray_and_blurred = blur_generator(i, 5, gray)
            circles = cv2.HoughCircles(image=gray_and_blurred, method=cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=30, minRadius=10, maxRadius=100)
            if circles is not None:
                circles = np.round(circles[0, :]).astype(int)
                for x, y, r in circles:
                    cv2.circle(gray_and_blurred, center=(x, y), radius=r, color=(0, 0, 0))
                cv2.imshow('Hough circles', gray_and_blurred)
            else:
                print('error')

        cv2.waitKey(0)
        cv2.destroyAllWindows()



