from PIL import Image
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt

file_path = input("Copy and paste the image's file path: ").strip('"')
if not os.path.exists(file_path):
    print('An error occured')
else:
    src = cv2.imread(file_path)

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
            model_path_face_landmark = r'C:\Users\abhid\Downloads\projs\testing\models\face_landmarker.task'
            model_path_image_segmenter = r"C:\Users\abhid\Downloads\projs\testing\models\selfie_segmenter_landscape.tflite"
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            ImageSegmenter = mp.tasks.vision.ImageSegmenter
            ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            optionsFaceLandmark = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path_face_landmark),
                running_mode=VisionRunningMode.IMAGE
            )
            optionsImageSegmenter = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path_image_segmenter),
                running_mode=VisionRunningMode.IMAGE,
                output_category_mask=True,
                output_confidence_masks=True
            )

            imageHeight, imageWidth, channels = src.shape
            heightScale = 1080 / imageHeight
            widthScale = 1920 / imageWidth
            scale = min(heightScale, widthScale, 1)
            scaledWidth = imageWidth * scale
            scaledHeight = imageHeight * scale
            output = cv2.resize(src, (int(scaledWidth), int(scaledHeight)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

            with FaceLandmarker.create_from_options(optionsFaceLandmark) as landmarker:
                mp_image = mp.Image.create_from_file(file_path)
                facemarker_results = landmarker.detect(mp_image)

            for landmark in facemarker_results.face_landmarks[0]:
                x = int(landmark.x * scaledWidth)
                y = int(landmark.y * scaledHeight)
                cv2.circle(img=output, center=(x, y), radius=1, color=(0, 0, 255), thickness=-1)

            with ImageSegmenter.create_from_options(optionsImageSegmenter) as segmentor:
                segmented_result = segmentor.segment(mp_image)
                confidence_mask = segmented_result.confidence_masks[0]
                alpha = (confidence_mask).numpy_view()
                alpha = cv2.resize(alpha, (int(scaledWidth), int(scaledHeight)), interpolation=cv2.INTER_LINEAR)
                alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
                
                binary_mask = (alpha > 0.3).astype(np.uint8)
                cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1, dst=binary_mask)
                cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1, dst=binary_mask)
                alpha *= binary_mask

                alpha = alpha[:, :, np.newaxis]
                foreground = (output.astype(np.float32) * alpha).astype(np.uint8)
                background = (output.astype(np.float32) * (1 - alpha)).astype(np.uint8)
                
            cv2.imshow("Mask", alpha)
            cv2.imshow("Foreground", foreground)
            cv2.imshow("Background", background)
            cv2.waitKey(0)
            cv2.destroyAllWindows()