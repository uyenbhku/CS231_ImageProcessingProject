# import dependencies
import cv2  # connect webcam, process images
import mediapipe as mp  # holistic API
import time     # to calculate FPS
import numpy as np # processing
from pygame import mixer
if __name__ == '__main__':

    # SETUP MEDIAPIPE
    print('Setting up................')
    mp_drawing = mp.solutions.drawing_utils # help draw the detections
    mp_holistic = mp.solutions.holistic # a Holistic class object

    # GET REALTIME WEBCAM FEED
    print('Getting webcam feed.................')

    ## define a video capture object, 0 is the webcam
    ## by default, each frame has size (480x640) (height x width)
    start, end = 0, 0 # helper variables to calculate FPS
    cap = cv2.VideoCapture(0)
    height, width = 720, 1280
    cap.set(3, width)
    cap.set(4, height)

    # Load effect angel_wings
    angel_wings_animation = cv2.VideoCapture('output_angel_wings.mp4')
    angel_wings_frame_count = 0

    # Load effect heart_light
    love_light_animation = cv2.VideoCapture('resized_heart_light.mp4')
    love_light_frame_count = 0

    # Sound
    mixer.init()
    mixer.music.load('magic_song.mp3')
    # To play sound
    count_sound = 1
    # Initiate holistic model
    print('Initiate Holistic Model') 
    with mp_holistic.Holistic( \
                enable_segmentation = True, \
                min_detection_confidence=0.5, \
                min_tracking_confidence=0.5) as holistic:
        print('Opening webcam feed........... Press q to stop')
        while cap.isOpened():

            # start = time.time()
            # Capture the video frame
            # by frame
            success, frame = cap.read()
            if not success:
                print('Cannot receive frame from camera')
                break

            # flip the image vertically for later selfie view display
            # recolor feed from BGR to RGB
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # to improve performance, mark the image as not writeable to
            # pass by reference instead of making a copy
            frame.flags.writeable = False
            
            # make detection
            results = holistic.process(frame) # store all different kinds of landmarks...

            # enable drawing landmark annotation on the frame
            frame.flags.writeable = True 
            # frame = np.zeros(frame.shape)
            # recolor feed from RGB to BGR so it can be displayed
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ### HIEU UNG DOI CANH THIEN THAN ###
            #----------------------------------------------------------------------------------------#
            if results.pose_landmarks:
                poseList = []
                myPose = results.pose_landmarks.landmark
                cx, cy = int(myPose[23].y * height), int(myPose[23].x * width)
                poseList.append([cx, cy])
                cx, cy = int(myPose[11].y * height), int(myPose[11].x * width)
                poseList.append([cx, cy])
                cx, cy = int(myPose[12].y * height), int(myPose[12].x * width)
                poseList.append([cx, cy])
                cx, cy = int(myPose[13].y * height), int(myPose[13].x * width)
                poseList.append([cx, cy])
                cx, cy = int(myPose[14].y * height), int(myPose[14].x * width)
                poseList.append([cx, cy])
                cx, cy = int(myPose[15].y * height), int(myPose[15].x * width)
                poseList.append([cx, cy])
                cx, cy = int(myPose[16].y * height), int(myPose[16].x * width)
                poseList.append([cx, cy])
                cx, cy = int(myPose[24].y * height), int(myPose[24].x * width)
                poseList.append([cx, cy])

                # Check dieu kien tay gio len và nguoi thang dung
                if ( # Hai tay gio len
                    poseList[2][0] > poseList[4][0]
                    and poseList[4][0] > poseList[6][0]
                    and poseList[1][0] > poseList[3][0]
                    and poseList[3][0] > poseList[5][0]
                    # Nguoi thang dung
                    and poseList[2][0] < poseList[7][0]
                    and poseList[1][0] < poseList[0][0]):

                    # Play sound
                    if (count_sound == 1):
                        mixer.music.play()
                        count_sound = 0

                    # Read a frame from angel_wings animation video
                    _, angel_wings_frame = angel_wings_animation.read()
    
                    # Increment the angel_wings animation video frame counter.
                    angel_wings_frame_count += 1
    
                    # Check if the current frame is the last frame of the smoke animation video.
                    if angel_wings_frame_count == angel_wings_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
        
                        # Set the current frame position to first frame to restart the video.
                        angel_wings_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
                        # Set the smoke animation video frame counter to zero.
                        angel_wings_frame_count = 0
                    
                    # Read a frame from love_light animation video
                    _, love_light_frame = love_light_animation.read()
                    # love_light_frame = cv2.resize(love_light_frame, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)
                    # Increment the love_light animation video frame counter.
                    love_light_frame_count += 1
    
                    # Check if the current frame is the last frame of the smoke animation video.
                    if love_light_frame_count == love_light_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
        
                        # Set the current frame position to first frame to restart the video.
                        love_light_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
                        # Set the smoke animation video frame counter to zero.
                        love_light_frame_count = 0

                    # Calculate size of wings
                    aw_ratio = angel_wings_frame.shape[0] / angel_wings_frame.shape[1]

                    width1 = np.sqrt((poseList[1][0] - poseList[2][0])**2 + (poseList[1][1] - poseList[2][1])**2)
                    width2 = np.sqrt((poseList[0][0] - poseList[7][0])**2 + (poseList[0][1] - poseList[7][1])**2)
                    width_body = width1 + width2

                    width_wing = width_body * 4

                    height_wing = aw_ratio * width_wing

                    # Calculate position to add effect
                    mid_top_x = (poseList[1][0] + poseList[2][0]) / 2
                    mid_top_y = (poseList[1][1] + poseList[2][1]) / 2

                    mid_bot_x = (poseList[0][0] + poseList[7][0]) / 2
                    mid_bot_y = (poseList[0][1] + poseList[7][1]) / 2

                    angle = np.arctan2(mid_top_x - mid_bot_x, mid_top_y - mid_bot_y)

                    length_center_bot = np.sqrt((mid_top_x - mid_bot_x)**2 + (mid_top_y - mid_bot_y)**2) * 2/3

                    center_x = mid_bot_x + np.sin(angle)*length_center_bot
                    center_y = mid_bot_y + np.cos(angle)*length_center_bot

                    mid_top_wing_x = center_x + np.sin(angle)*height_wing/2                      
                    mid_top_wing_y = center_y + np.cos(angle)*height_wing/2

                    mid_bot_wing_x = center_x - np.sin(angle)*height_wing/2                       
                    mid_bot_wing_y = center_y - np.cos(angle)*height_wing/2

                    angle_top = np.arctan2(poseList[1][0] - poseList[2][0], poseList[1][1] - poseList[2][1])
                    angle_bot = np.arctan2(poseList[0][0] - poseList[7][0], poseList[0][1] - poseList[7][1])

                    left_top_wing_x = int(mid_top_wing_x - np.sin(angle_top)*width_wing/2)
                    left_top_wing_y = int(mid_top_wing_y - np.cos(angle_top)*width_wing/2)

                    right_top_wing_x = int(mid_top_wing_x + np.sin(angle_top)*width_wing/2)
                    right_top_wing_y = int(mid_top_wing_y + np.cos(angle_top)*width_wing/2)

                    left_bot_wing_x = int(mid_bot_wing_x - np.sin(angle_bot)*width_wing/2)
                    left_bot_wing_y = int(mid_bot_wing_y - np.cos(angle_bot)*width_wing/2)

                    right_bot_wing_x = int(mid_bot_wing_x + np.sin(angle_bot)*width_wing/2)
                    right_bot_wing_y = int(mid_bot_wing_y + np.cos(angle_bot)*width_wing/2)

                    src_points = np.float32([[0, 0], [angel_wings_frame.shape[1], 0], [0, angel_wings_frame.shape[0]], [angel_wings_frame.shape[1], angel_wings_frame.shape[0]]])
                    dst_points = np.float32([[left_top_wing_y, left_top_wing_x],
                         [right_top_wing_y, right_top_wing_x],
                         [left_bot_wing_y, left_bot_wing_x],
                         [right_bot_wing_y, right_bot_wing_x]])

                    # Compute the perspective transformation matrix
                    M = cv2.getPerspectiveTransform(src_points, dst_points)

                    # Apply the perspective transformation
                    angel_wings_frame = cv2.warpPerspective(angel_wings_frame, M, (frame.shape[1], frame.shape[0]))

                    mask_wing = angel_wings_frame[:,:,0] + angel_wings_frame[:,:,1] + angel_wings_frame[:,:,2]
                    
                    # Nguong de ap dung hieu ung wing la 10
                    if (poseList[1][1] > poseList[2][1]):
                        frame_temp = frame.copy()

                        frame[mask_wing[:,:] > 10] = angel_wings_frame[mask_wing[:,:] > 10]
                    
                        frame[results.segmentation_mask[:,:] > 0.5] = frame_temp[results.segmentation_mask[:,:] > 0.5]

                        frame[results.segmentation_mask[:,:] > 0.5] = 0.6*frame[results.segmentation_mask[:,:] > 0.5] + \
                            0.4*love_light_frame[results.segmentation_mask[:,:] > 0.5]
                    elif (poseList[1][1] <= poseList[2][1]):
                        frame[results.segmentation_mask[:,:] > 0.5] = 0.6*frame[results.segmentation_mask[:,:] > 0.5] + \
                            0.4*love_light_frame[results.segmentation_mask[:,:] > 0.5]
                        
                        frame[mask_wing[:,:] > 10] = angel_wings_frame[mask_wing[:,:] > 10]
                else:
                    mixer.music.stop()
                    count_sound = 1
            #----------------------------------------------------------------------------------------#
            
            # calculate how long this code takes to process a frame on a CPU
            # end = time.time()  
            # fps = 1/(end - start)
            # display FPS on the frame
            # cv2.putText(frame, str(f'FPS: {int(fps)}'), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 3)
            
            # Display the resulting frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            cv2.imshow('Webcam Feed', frame)
            #cv2.waitKey(0)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()