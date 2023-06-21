# import dependencies
import cv2  # connect webcam, process images
import mediapipe as mp  # holistic API
import time     # to calculate FPS
# import tensorflow as tf, keras  # to create model for training
# from keras.models import load_model # load pretrained model
import numpy as np # processing
import sys
from pygame import mixer



def resize_image(image, size=(640, 480)):

    if image.shape[0] > size[1] or image.shape[1] > size[0]:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # expand if smaller
    elif image.shape[0] < size[1] or image.shape[1] < size[0]:
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
            asset = resize_image(asset, window_size)
        asset = cv2.cvtColor(asset, cv2.COLOR_BGR2RGB)

    elif asset_extension in ('gif', 'mp4', 'webm', 'mkv', 'mpeg'):
        asset_gif = cv2.VideoCapture(path)
        while asset_gif.isOpened():
            is_success, f = asset_gif.read()
            if not is_success: 
                break
            # print(f.shape)
            if window_size:
                f = resize_image(f, window_size)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            asset.append(f)
            
        asset = np.array(asset)

    else:
        print(f'unidentified file format, path = {path}')
        # early stop the programme
        return None

    return asset



def cal_euclidean_distance(lm1, lm2):
    return np.sqrt(np.power(lm1[0] - lm2[0], 2) + np.power(lm1[1] - lm2[1], 2))


def mouth_gif():

    width, height = 1280, 720
    print('Reading mouth effect.................')
    # read effect
    effect_path = '.\\assets\\background\\neon_hearts.mp4'
    effect = read_asset(effect_path, (width, height))
    if effect is None: 
        print('Cannot read mouth effect')
        sys.exit(0)
    mouth_effect_count = 0
    no_eff_frames = effect.shape[0]
    # define effect opacity
    alpha = 0.25

    print('Reading mouth gif.................')
    # read gif frames
    gif_path = '.\\assets\\gifs\\ezgif.com-crop.gif'
    gif_frames = read_asset(gif_path)
    if gif_frames is None: 
        print('Cannot read mouth gif')
        sys.exit(0)
    
    print('Reading mouth audio..........')
    audio_path = './audio/ily_voice.mp3'
    #Instantiate mixer
    mixer.init()
    # load audio
    mixer.music.load(audio_path)
    # play the music 
    mixer.music.play(-1)
    # pause 
    mixer.music.pause()

    no_frames = gif_frames.shape[0]
    mouth_gif_count = 0

    # GET REALTIME WEBCAM FEED
    print('Getting webcam feed.................')
    ## define a video capture object, 0 is the webcam
    ## by default, each frame has size (480x640) (height x width)
    start, end = 0, 0 # helper variables to calculate FPS
    demo_path = '.\\assets\\demo\\womanyell.mp4'
    cap = cv2.VideoCapture(demo_path)
    cap.set(3, width)
    cap.set(4, height)

    # Set up mediapipe
    print('Setting up model..........')
    print('Initiate Holistic Model') 
    # Initiate holistic model
    mp_holistic = mp.solutions.holistic # a Holistic class object

    OPENED_THRESHOLD = 125 # DEGREES
    PI = 57.2958
    tightness = .3
    
    print('Opening webcam feed........... Press ESC to stop')
    with mp_holistic.Holistic( \
                                # model_complexity=2,
                                enable_segmentation=True,
                                min_detection_confidence=0.5, \
                                min_tracking_confidence=0.5) as holistic:
        # mask = None
        while cap.isOpened():
            start = time.time()
            # Capture the video frame by frame
            success, frame = cap.read()
            
            if not success:
                print('Cannot receive frame from camera')
                break
            # PREPROCESS
            frame = resize_image(frame, (width, height))
            
            # flip the image vertically for later selfie view display
            # recolor feed from BGR to RGB so that the model will have good performance
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # to improve performance, mark the image as not writeable to
            # pass by reference instead of making a copy
            frame.flags.writeable = False


            # detect location
            locations = holistic.process(frame) # store all different kinds of landmarks...
            
            # PROCESS
            frame.flags.writeable = True 

            mouth_landmarks = []
            #### mouth filter
            
            # check if mouth is detected in the frame
            if locations.face_landmarks:
                
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[61].x * width, locations.face_landmarks.landmark[61].y * height]))
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[291].x * width, locations.face_landmarks.landmark[291].y * height]))
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[0].x * width, locations.face_landmarks.landmark[0].y * height]))
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[17].x * width, locations.face_landmarks.landmark[17].y * height]))

                ## calculate the mouth area
                mouth_width = int(cal_euclidean_distance(mouth_landmarks[0], mouth_landmarks[1]))
                mouth_height = int(cal_euclidean_distance(mouth_landmarks[2], mouth_landmarks[3]))

                ## check if the mouth is opened 
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[14].x * width, locations.face_landmarks.landmark[14].y * height]))
                vector_1 = mouth_landmarks[4] - mouth_landmarks[1]
                vector_2 = mouth_landmarks[4] - mouth_landmarks[0]
                angle = np.arccos(np.dot(vector_1 / np.linalg.norm(vector_1), \
                                                    vector_2 / np.linalg.norm(vector_2))) * PI
                
                if angle < OPENED_THRESHOLD: 

                    mask = np.zeros((height, width), dtype='uint8')
                    mask[locations.segmentation_mask > tightness] = 255
                    face = cv2.bitwise_and(frame, frame, mask=mask)

                    # Find Canny edges
                    edged = cv2.Canny(mask, 30, 200)
                    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    face = cv2.drawContours(face, contours, -1, (255, 255, 255), 10)

                    # blend background
                    bg_img = frame.copy()
                    # Create the overlay
                    frame[mask == 0] = cv2.addWeighted(bg_img, 1-alpha, effect[mouth_effect_count%no_eff_frames], alpha, 0.0)[mask == 0]
                    
                    blurred_bg = cv2.GaussianBlur(frame, (21, 21), 0)
                    frame[face == (255, 255, 255)] = blurred_bg[face == (255, 255, 255)]

                    gif_frame = gif_frames[mouth_gif_count%no_frames]
                    ### resize the gif frame to fit the mouth area
                    ## resize
                    gif_frame = resize_image(gif_frame, (mouth_width, mouth_height))
                    
                    ### translate the gif frame
                    ## calculate the coords to translate to 
                    tx = mouth_landmarks[0][0]
                    ty = mouth_landmarks[2][1]
                    ## init a translation matrix
                    translation_matrix = np.array([
                        [1, 0, tx],
                        [0, 1, ty]
                    ], dtype=np.float32)
                    ## translate to the calculated area
                    gif_frame = cv2.warpAffine(src=gif_frame, M=translation_matrix, \
                                                dsize=(width, height), \
                                                borderMode=cv2.BORDER_REPLICATE) 
                    
                    center_x = int(abs(mouth_landmarks[0][0] + mouth_landmarks[1][0]) / 2)
                    center_y = int(abs(mouth_landmarks[2][1] + mouth_landmarks[3][1]) / 2)
                    # rotate gif 
                    vector_1 = [int((- mouth_landmarks[0][0] + mouth_landmarks[1][0])), \
                                    int((- mouth_landmarks[0][1] + mouth_landmarks[1][1]))]

                    vector_2 = [1, 0]
                    angle = np.arccos(np.dot(vector_1 / np.linalg.norm(vector_1), \
                                                vector_2 / np.linalg.norm(vector_2)))*57.2958
                    if (mouth_landmarks[0][1] < mouth_landmarks[1][1]): angle *= -1

                    rotate_matrix = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=angle, scale=1.5)
                    gif_frame = cv2.warpAffine(src=gif_frame, M=rotate_matrix, \
                                                dsize=(width, height), \
                                                borderMode=cv2.BORDER_REPLICATE)

                    # add animation to the frame
                    gray_gif = cv2.cvtColor(gif_frame, cv2.COLOR_RGB2GRAY)
                    gray_gif[gray_gif < 10] = 255
                    frame[gray_gif < 255] = cv2.addWeighted(frame, 0, gif_frame, 1, 0.0)[gray_gif < 255]
                    # Resume the music
                    mixer.music.unpause()
                else:
                    mixer.music.pause()
                    mixer.music.set_pos(0)
            
            # move to next frame of the gif
            mouth_gif_count += 1
            mouth_effect_count += 1

            # recolor feed from RGB to BGR so it can be displayed by openCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # calculate how long this code takes to process a frame on a CPU
            end = time.time()  
            fps = 1/(end - start)
            # display FPS on the frame
            cv2.putText(frame, str(f'FPS: {int(fps)}'), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)
            
            # Display the resulting frame
            cv2.imshow('Webcam Feed', frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # press ESC to terminate 
                break

    # After the loop release the cap object
    cap.release()
    
    # Destroy all the windows
    cv2.destroyAllWindows()



if __name__ == '__main__':
    mouth_gif()