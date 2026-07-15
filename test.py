from PIL import Image
import os
import cv2
import numpy as np
import mediapipe as mp

file_path = input("Copy and paste the image's file path: ").strip('"')
if not os.path.exists(file_path):
    print('An error occured')
else:
    src = cv2.imread(file_path)
    print(src.shape)

    if src is None:
        print('OpenCV image opening error')
    else:
        appFunction = input('Select an option from the following: Image filtration, Facial animator...: ').lower()

        if appFunction == 'image filtration':
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
                blurred_image = blur_generator(1, maxKernalLength, src)
                cv2.imshow('Blurred image', blurred_image)
            elif userChoice == 'hough circles':
                gray_and_blurred = blur_generator(1, 5, gray)
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

        elif appFunction == 'facial animator':
            model_path = r'C:\Users\abhid\Downloads\projs\testing\models\face_landmarker.task'
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE
            )

            output = src.copy()

            with FaceLandmarker.create_from_options(options) as landmarker:
                mp_image = mp.Image.create_from_file(file_path)
                facemarker_results = landmarker.detect(mp_image)

            imageHeight, imageWidth, channels = src.shape
            for landmark in facemarker_results.face_landmarks[0]:
                x = int(landmark.x * imageWidth)
                y = int(landmark.y * imageHeight)
                cv2.circle(img=output, center=(x, y), radius=1, color=(0, 0, 255), thickness=-1)
            
            print(output.shape)
            
            cv2.namedWindow("Face Landmarks", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Face Landmarks", (imageHeight, imageWidth))
            cv2.imshow("Face Landmarks", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()