# import dependencies
import cv2 # connect webcam, process images
import mediapipe as mp # holistic API
import time # to calculate FPS
import numpy as np # processing

# SETUP MEDIAPIPE
print('Setting up................')
mp_drawing = mp.solutions.drawing_utils # help draw the detections
mp_holistic = mp.solutions.holistic # a Holistic class object
mp_drawing_styles = mp.solutions.drawing_styles

# GET REALTIME WEBCAM FEED
print('Getting webcam feed.................')
## define a video capture object, 0 is the webcam
start, end = 0, 0 # helper variables to calculate FPS
url = "http://10.0.23.140:8080/video"
cap = cv2.VideoCapture(url)
effect = cv2.VideoCapture('eff.gif')
right_count_1 = 0
another_1 = cv2.VideoCapture('morning.gif')
another_2 = cv2.VideoCapture('magic.gif')
another_3 = cv2.VideoCapture('pink.gif')
right_count_2 = 0 
left_count_1 = 0
left_count_2 = 0
right_left_count_1 = 0
right_left_count_2 = 0
print('Initiate Holistic Model')

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as holistic:
    print('Opening webcam feed........... Press q to stop')
    while cap.isOpened():
        start = time.time()
        # Capture the video frame by frame
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (1280, 720))
        # to improve performance, mark the image as not writeable to pass by reference instead of making a copy
        frame.flags.writeable = False
        # make detection
        # store all different kinds of landmarks...
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # enable drawing landmark annotation on the frame
        frame.flags.writeable = True 
        # get frame shape
        fr_w, fr_h, fr_c = frame.shape
        # blending variable
        alpha = 0
        beta = 0.1
        gamma = 0.6

        if results.left_hand_landmarks and results.right_hand_landmarks:
            if abs(results.left_hand_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[6].x) < 0.03 and abs(results.left_hand_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[6].y) < 0.03 and abs(results.right_hand_landmarks.landmark[3].x - results.right_hand_landmarks.landmark[6].x) < 0.03 and abs(results.right_hand_landmarks.landmark[3].y - results.right_hand_landmarks.landmark[6].y) < 0.03:
            
                right_left_count_1+=1
                if right_left_count_1 == effect.get(cv2.CAP_PROP_FRAME_COUNT):
                    effect.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    right_left_count_1 = 0
                right_left_count_2+=1
                if right_left_count_2 == another_3.get(cv2.CAP_PROP_FRAME_COUNT):
                    another_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    right_left_count_2 = 0

                okay, bg = effect.read()
                if not okay:
                    continue
                ok, eff = another_3.read()
                if not ok:
                    continue
                w, h, c = bg.shape
                e_w, e_h, e_c = eff.shape
                eff = cv2.resize(eff, (fr_h, fr_w))
                x = int(results.left_hand_landmarks.landmark[3].x *fr_h)
                y = int(results.left_hand_landmarks.landmark[3].y *fr_w)
                z = int(results.right_hand_landmarks.landmark[3].x *fr_h)
                t = int(results.right_hand_landmarks.landmark[3].y *fr_w)
                
                translation_matrix_1 = np.array([ [1, 0, int(x-h/2)] ,[0, 1, int(y-w/2)]], dtype=np.float32)
                translated_bg_1 = cv2.warpAffine(src=bg, M=translation_matrix_1, dsize=(fr_h, fr_w))
                translation_matrix_2 = np.array([ [1, 0, int(z-h/2)] ,[0, 1, int(t-w/2)]], dtype=np.float32)
                translated_bg_2 = cv2.warpAffine(src=bg, M=translation_matrix_2, dsize=(fr_h, fr_w))
                gif_1 = cv2.cvtColor(translated_bg_1, cv2.COLOR_BGR2GRAY)
                gif_2 = cv2.cvtColor(translated_bg_2, cv2.COLOR_BGR2GRAY)
                mask = results.segmentation_mask.copy()
                mask[mask <= 0.75] = 0
                mask = (mask*255).astype('uint8')
                edges = cv2.Canny(mask,200,200)
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame[mask < 0.05] = frame[mask < 0.05] * beta + eff[mask < 0.05] * (1 - beta)
                blur = cv2.GaussianBlur(frame, (21, 21), 0)
                cv2.drawContours(mask, contours, -1, (100), 10)
                frame[mask==100] = blur[mask==100]
                gif_1[gif_1==0] = 255
                frame[gif_1<255] = cv2.addWeighted(frame, alpha, translated_bg_1, 1 - alpha, 0)[gif_1<255]
                gif_2[gif_2==0] = 255
                frame[gif_2<255] = cv2.addWeighted(frame, alpha, translated_bg_2, 1 - alpha, 0)[gif_2<255]

                # mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        if results.left_hand_landmarks and results.right_hand_landmarks is None:
            if abs(results.left_hand_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[6].x) < 0.03 and abs(results.left_hand_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[6].y) < 0.03:
                # alpha = np.multiply(alpha, 1.0/255)
                left_count_1+=1
                if left_count_1 == effect.get(cv2.CAP_PROP_FRAME_COUNT):
                    effect.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    left_count_1 = 0
                left_count_2+=1
                if left_count_2 == another_1.get(cv2.CAP_PROP_FRAME_COUNT):
                    another_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    left_count_2 = 0

                okay, bg = effect.read()
                if not okay:
                    continue
                _, eff = another_1.read()
                
                w, h, c = bg.shape
                e_w, e_h, e_c = eff.shape
                eff = cv2.resize(eff, (fr_h, fr_w))
                x = int(results.left_hand_landmarks.landmark[3].x *fr_h)
                y = int(results.left_hand_landmarks.landmark[3].y *fr_w)

                # print(results.left_hand_landmarks.landmark[3].z)
                translation_matrix = np.array([ [1, 0, int(x-h/2)] ,[0, 1, int(y-w/2)]], dtype=np.float32)
                translated_bg = cv2.warpAffine(src=bg, M=translation_matrix, dsize=(fr_h, fr_w))
                gif = cv2.cvtColor(translated_bg, cv2.COLOR_BGR2GRAY)
                mask = results.segmentation_mask.copy()
                mask[mask <= 0.75] = 0
                mask = (mask*255).astype('uint8')
                edges = cv2.Canny(mask,200,200)
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame[mask < 0.05] = frame[mask < 0.05] * beta + eff[mask < 0.05] * (1 - beta)
                blur = cv2.GaussianBlur(frame, (21, 21), 0)
                cv2.drawContours(mask, contours, -1, (100), 10)
                frame[mask==100] = blur[mask==100]
                gif[gif==0] = 255
                frame[gif<255] = cv2.addWeighted(frame, alpha, translated_bg, 1 - alpha, 0)[gif<255]
                    
            # mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
        if results.right_hand_landmarks and results.left_hand_landmarks is None:
            # thumb = [results.left_hand_landmarks.landmark[1].x - results.left_hand_landmarks.landmark[4].x, results.left_hand_landmarks.landmark[1].y - results.left_hand_landmarks.landmark[4].y]
            # index = [results.left_hand_landmarks.landmark[5].x - results.left_hand_landmarks.landmark[8].x, results.left_hand_landmarks.landmark[5].y - results.left_hand_landmarks.landmark[8].y]
            if abs(results.right_hand_landmarks.landmark[3].x - results.right_hand_landmarks.landmark[6].x) < 0.03 and abs(results.right_hand_landmarks.landmark[3].y - results.right_hand_landmarks.landmark[6].y) < 0.03:
                # alpha = np.multiply(alpha, 1.0/255)
                right_count_1+=1
                if right_count_1 == effect.get(cv2.CAP_PROP_FRAME_COUNT):
                    effect.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    right_count_1 = 0
                right_count_2+=1
                if right_count_2 == another_2.get(cv2.CAP_PROP_FRAME_COUNT):
                    another_2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    right_count_2 = 0
                okay, bg = effect.read()
                if not okay:
                    continue
                _, eff = another_2.read()
            
                w, h, c = bg.shape
                e_w, e_h, e_c = eff.shape

                eff = cv2.resize(eff, (fr_h, fr_w))
                x = int(results.right_hand_landmarks.landmark[3].x *fr_h)
                y = int(results.right_hand_landmarks.landmark[3].y *fr_w)
                translation_matrix = np.array([ [1, 0, int(x-h/2)] ,[0, 1, int(y-w/2)]], dtype=np.float32)
                translated_bg = cv2.warpAffine(src=bg, M=translation_matrix, dsize=(fr_h, fr_w))
                gif = cv2.cvtColor(translated_bg, cv2.COLOR_BGR2GRAY)
                mask = results.segmentation_mask.copy()
                mask[mask <= 0.75] = 0
                mask = (mask*255).astype('uint8')
                edges = cv2.Canny(mask,200,200)
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                blend = frame[int(y-w/2):int(y+w/2), int(x-h/2):int(x+h/2), :]
                frame[mask != 0] = frame[mask != 0] * gamma + eff[mask != 0] * (1 - gamma)
                blur = cv2.GaussianBlur(frame, (21, 21), 0)
                cv2.drawContours(mask, contours, -1, (100), 10)
                frame[mask==100] = blur[mask==100]
                gif[gif==0] = 255
                frame[gif<255] = cv2.addWeighted(frame, alpha, translated_bg, 1 - alpha, 0)[gif<255]

            # mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
        # calculate how long this code takes to process a frame on a CPU
        end = time.time()  
        fps = 1/(end - start)

        # display FPS on the frame
        cv2.putText(frame, str(f'FPS: {int(fps)}'), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        #resize window screen
        cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Webcam Feed', 1280, 720)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()