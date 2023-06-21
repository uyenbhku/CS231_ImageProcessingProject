# import dependencies
import cv2  # connect webcam, process images
import mediapipe as mp  # holistic API
import time     # to calculate FPS
# import tensorflow as tf, keras  # to create model for training
# from keras.models import load_model # load pretrained model
import numpy as np # processing

import imageio



if __name__ == '__main__':
    width, height = 1280, 720
    # SETUP MEDIAPIPE
    print('Setting up................')
    mp_drawing = mp.solutions.drawing_utils # help draw the detections
    mp_holistic = mp.solutions.holistic # a Holistic class object

    # GET REALTIME WEBCAM FEED
    print('Getting webcam feed.................')
    ## define a video capture object, 0 is the webcam
    ## by default, each frame has size (480x640) (480 x 640)
    start, end = 0, 0 # helper variables to calculate FPS
    demo_path = '.\\assets\\demo\\mantalking.mp4'
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    print('Initiate Holistic Model') 
    # Initiate holistic model
    # dataset = []
    demo_frames = []
    with mp_holistic.Holistic( \
                # model_complexity=2,
                min_detection_confidence=0.5, \
                min_tracking_confidence=0.5) as holistic:
        print('Opening webcam feed........... Press q to stop')
        while cap.isOpened():

            start = time.time()
            # Capture the video frame
            # by frame
            success, frame = cap.read()
            if not success:
                print('Cannot receive frame from camera')
                break

            # flip the image vertically for later selfie view display
            # recolor feed from BGR to RGB so that the model will have good performance
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # to improve performance, mark the image as not writeable to
            # pass by reference instead of making a copy
            frame.flags.writeable = False
            
            # make detection
            results = holistic.process(frame) # store all different kinds of landmarks...

            # enable drawing landmark annotation on the frame
            frame.flags.writeable = True 
            cv2.imshow('ko che', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame = np.zeros(frame.shape) 
            # recolor feed from RGB to BGR so it can be displayed
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # extract mouth features
            # mouth_landmarks = []
            # if results.face_landmarks:
            #     mouth_landmarks.append(np.array([results.face_landmarks.landmark[61].x * width, results.face_landmarks.landmark[61].y * height]))
            #     mouth_landmarks.append(np.array([results.face_landmarks.landmark[291].x * width, results.face_landmarks.landmark[291].y * height]))
            #     mouth_landmarks.append(np.array([results.face_landmarks.landmark[0].x * width, results.face_landmarks.landmark[0].y * height]))
            #     mouth_landmarks.append(np.array([results.face_landmarks.landmark[17].x * width, results.face_landmarks.landmark[17].y * height]))

            #     mouth_landmarks.append(np.array([results.face_landmarks.landmark[14].x * width, results.face_landmarks.landmark[14].y * height]))
            #     # mouth_landmarks.append(np.array([results.face_landmarks.landmark[87].x * width, results.face_landmarks.landmark[87].y * height]))
            #     # mouth_landmarks.append(np.array([results.face_landmarks.landmark[312].x * width, results.face_landmarks.landmark[312].y * height]))
            #     # mouth_landmarks.append(np.array([results.face_landmarks.landmark[317].x * width, results.face_landmarks.landmark[317].y * height]))
            #     vector_1 = mouth_landmarks[4] - mouth_landmarks[1]
            #     vector_2 = mouth_landmarks[4] - mouth_landmarks[0]
            #     angle = np.arccos(np.dot(vector_1 / np.linalg.norm(vector_1), \
            #                                         vector_2 / np.linalg.norm(vector_2)))*57.2958
            #     for x, y in mouth_landmarks:
            #         frame = cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=5)
            #     # draw facemesh 
            #     mp_drawing.draw_landmarks(frame, \
            #                             results.face_landmarks,\
            #                             mp_holistic.FACEMESH_TESSELATION,\
            #                             # stylizing
            #                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1), 
            #                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))
            #     cv2.putText(frame, str(f'angle: {round(angle, 3)} degree'), (10, 680), cv2.FONT_HERSHEY_COMPLEX, 3, (153,43,37), 3)
            # # draw pose landmarks
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            # draw left hand landmarks
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # if results.left_hand_landmarks: print(results.left_hand_landmarks)
            # draw right hand landmarks
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # calculate how long this code takes to process a frame on a CPU
            end = time.time()  
            fps = 1/(end - start)
            # display FPS on the frame
            # cv2.putText(frame, str(f'FPS: {int(fps)}'), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)
            
            # demo_frames.append(frame)
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
    # imageio.mimsave('./proof/mouth_angle.gif', demo_frames, fps=10)
    print('Saving mouth dataset..........')
    # print(dataset)
