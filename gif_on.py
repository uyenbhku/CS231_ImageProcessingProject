# import dependencies
import cv2  # connect webcam, process images
# import mediapipe as mp  # holistic API
import time     # to calculate FPS
# import tensorflow as tf, keras  # to create model for training
# from keras.models import load_model # load pretrained model
import numpy as np # processing
import sys



def resize_image(image, size=(640, 480)):

    if image.shape[0] > 480 or image.shape[1] > 640:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # expand if smaller
    elif image.shape[0] < 480 or image.shape[1] < 640:
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    return image




def read_asset(path, window_size=None):
    # read effect
    asset_extension = path.strip().rsplit('.', maxsplit=1)[1]
    asset = []
    if asset_extension in ('jpg', 'png', 'jpeg'):
        asset = cv2.imread(path)
        # auto resize 
        if window_size:
            asset = resize_image(asset)
        asset = cv2.cvtColor(asset, cv2.COLOR_BGR2RGB)

    elif asset_extension in ('gif', 'mp4'):
        asset_gif = cv2.VideoCapture(path)
        while asset_gif.isOpened():
            is_success, f = asset_gif.read()
            if not is_success: 
                break
            if window_size:
                f = resize_image(f)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            asset.append(f)
            
        asset = np.array(asset)

    else:
        print(f'unidentified file format, path = {path}')
        # early stop the programme
        return None

    return asset


if __name__ == '__main__':

    print('Reading effect.................')
    # read effect
    effect_path = 'https://raw.githubusercontent.com/uyenbhku/CS231_ImageProcessingProject/main/assets/background/360_F_322897394_X2JQen9I6ECDSsQZoESOJ87dfebaGaAe.jpg'
    effect = read_asset(effect_path, (640, 480))
    if effect is None: 
        sys.exit(0)
         
    # define effect opacity
    alpha = 0.3

    print('Reading gif.................')
    # read gif frames
    gif_path = 'https://raw.githubusercontent.com/uyenbhku/CS231_ImageProcessingProject/main/assets/demo/Y1bY.gif'
    gif_frames = read_asset(gif_path)
    if gif_frames is None: 
        sys.exit(0)

    no_frames = gif_frames.shape[0]
    gif_w, gif_h = gif_frames[0].shape[0], gif_frames[0].shape[1]
    count = 0

    # GET REALTIME WEBCAM FEED
    print('Getting webcam feed.................')
    ## define a video capture object, 0 is the webcam
    ## by default, each frame has size (480x640) (height x width)
    start, end = 0, 0 # helper variables to calculate FPS
    cap = cv2.VideoCapture(0)

    # Set up mediapipe
    print('Setting up model..........')
    # mp_drawing = mp.solutions.drawing_utils # help draw the detections
    print('Initiate Holistic Model') 
    # Initiate holistic model
    # mp_holistic = mp.solutions.holistic # a Holistic class object
    # dataset = []

    # UNCOMMENT THIS TO ENABLE SKIN SEGMENTATION
    # # Initiate selfie segmentation
    # print('Initiate Segmentation Model') 
    # mp_selfie_segmentation = mp.solutions.selfie_segmentation
    # BG_COLOR = (0, 0, 0) # black

    speed = 10
    opacity = list(range(no_frames * speed)) + list(range(no_frames * speed - 2, -1, -1))
    opacity = np.array(opacity) / (no_frames*speed)
    turn_on_filter = 0 ## 0: filter turns off, 1 on
    
    print('Opening webcam feed........... Press ESC to stop')
    # UNCOMMENT THIS TO ENABLE SKIN SEGMENTATION, IF YOU WANT TO USE BOTH DETECT POSE AND SEGMENTATION, INDENT ALL BELOW LINES OF CODE ONE MORE LEVEL
    # with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    # with mp_holistic.Holistic( \
    #                             # model_complexity=2,
    #                             min_detection_confidence=0.5, \
    #                             min_tracking_confidence=0.5) as holistic:
        # mask = None
    while cap.isOpened():

        start = time.time()
        # Capture the video frame by frame
        success, frame = cap.read()
        
        if not success:
            print('Cannot receive frame from camera')
            break
        
        # PREPROCESS
        # flip the image vertically for later selfie view display
        # recolor feed from BGR to RGB so that the model will have good performance
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # to improve performance, mark the image as not writeable to
        # pass by reference instead of making a copy
        frame.flags.writeable = False

        # segment selfie image => get mask
        # mask = (selfie_segmentation.process(frame).segmentation_mask * 255).astype('uint8')

        # detect location
        # locations = holistic.process(frame) # store all different kinds of landmarks...

        # PROCESS
        frame.flags.writeable = True 
        # face = cv2.bitwise_and(frame, frame, mask=mask)

        # # blending ----- UNCOMMENT THIS TO ENABLE HUMAN SEGMENTATION
        # blended_frame = frame.copy()
        # frame[mask < 2] = frame[mask < 2] * (1-alpha) + effect[mask < 2] * (alpha)

    
        # add animation to the frame
        for dim in range(3):
            frame[:gif_w, :gif_h, dim][gif_frames[count%no_frames][:, :, dim] < 255] = frame[:gif_w, :gif_h, dim][gif_frames[count%no_frames][:, :, dim] < 255]*(1-opacity[count])+ opacity[count] * gif_frames[count%no_frames][:, :, dim][gif_frames[count%no_frames][:, :, dim] < 255]

        # # draw facemesh 
        # mp_drawing.draw_landmarks(frame, \
        #                         locations.face_landmarks,\
        #                         mp_holistic.FACEMESH_TESSELATION,\
        #                         # stylizing
        #                         mp_drawing.DrawingSpec(color=(121,22,76), thickness=0, circle_radius=1),
        #                         mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2))
        # # draw pose landmarks
        # mp_drawing.draw_landmarks(frame, locations.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # # draw left hand landmarks
        # mp_drawing.draw_landmarks(frame, locations.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # # if results.left_hand_landmarks: print(results.left_hand_landmarks)
        # # draw right hand landmarks
        # mp_drawing.draw_landmarks(frame, locations.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # recolor feed from RGB to BGR so it can be displayed by openCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        

        # calculate how long this code takes to process a frame on a CPU
        end = time.time()  
        fps = 1/(end - start)
        # display FPS on the frame
        cv2.putText(frame, str(f'FPS: {int(fps)}'), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)
        
        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)
        # cv2.imshow('Mask', dst)

        if cv2.waitKey(1) & 0xFF == 27: # press ESC to terminate 
            break
        if cv2.waitKey(1) & 0xFF == ord('1'): # press 1 to turn on filter
            count = 1
            turn_on_filter = 1
        # move to next frame of the gif
        if count < no_frames*speed*2 - 2:
            count += 1
        elif count == no_frames*speed*2 - 1: 
            turn_on_filter = 0

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    # print('Saving mouth dataset..........')
    # print(dataset)

