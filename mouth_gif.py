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
    print('Reading effect.................')
    # read effect
    effect_path = '.\\assets\\background\\neon_hearts.mp4'
    effect = read_asset(effect_path, (width, height))
    if effect is None: 
        print('Cannot read effect')
        sys.exit(0)
    effect_count = 0
    no_eff_frames = effect.shape[0]
    # define effect opacity
    alpha = 0.25

    print('Reading gif.................')
    # read gif frames
    gif_path = '.\\assets\\gifs\\ezgif.com-crop.gif'
    gif_frames = read_asset(gif_path)
    if gif_frames is None: 
        print('Cannot read gif')
        sys.exit(0)
    
    print('Reading audio..........')
    audio_path = './audio/ily_voice.mp3'
    # audio = MediaPlayer(audio_path)
    #Instantiate mixer
    mixer.init()
    # load audio
    mixer.music.load(audio_path)
    # play the music 
    mixer.music.play(-1)
    # pause 
    mixer.music.pause()

    no_frames = gif_frames.shape[0]
    # gif_w, gif_h = gif_frames[0].shape[0], gif_frames[0].shape[1]
    gif_count = 0

    # GET REALTIME WEBCAM FEED
    print('Getting webcam feed.................')
    ## define a video capture object, 0 is the webcam
    ## by default, each frame has size (480x640) (height x width)
    start, end = 0, 0 # helper variables to calculate FPS
    demo_path = '.\\assets\\demo\\womanyell.mp4'
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    # Set up mediapipe
    print('Setting up model..........')
    # mp_drawing = mp.solutions.drawing_utils # help draw the detections
    print('Initiate Holistic Model') 
    # Initiate holistic model
    mp_holistic = mp.solutions.holistic # a Holistic class object
    # dataset = []

    # UNCOMMENT THIS TO ENABLE SKIN SEGMENTATION
    # Initiate selfie segmentation
    # print('Initiate Segmentation Model') 
    # mp_selfie_segmentation = mp.solutions.selfie_segmentation
    # BG_COLOR = (0, 0, 0) # black

    speed = 10
    # opacity = list(range(no_frames * speed)) + list(range(no_frames * speed - 2, -1, -1))
    opacity = list(range(no_frames * speed, -1, -1))
    # opacity_level = 31
    opacity = np.array(opacity) / (no_frames*speed)

    # ROI = None
    OPENED_THRESHOLD = 125 # DEGREES
    PI = 57.2958
    tightness = .3
    # cv2.imshow('Webcam Feed', effect[0])
    # cv2.waitKey(3000)

    # res = []
    print('Opening webcam feed........... Press ESC to stop')
    # UNCOMMENT THIS TO ENABLE SKIN SEGMENTATION, IF YOU WANT TO USE BOTH DETECT POSE AND SEGMENTATION, INDENT ALL BELOW LINES OF CODE ONE MORE LEVEL
    # with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
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
            # print(frame.shape)
            
            if not success:
                print('Cannot receive frame from camera')
                break
            # _, val = audio.get_frame(show=False)
            # if val == 'eof':
            #     break
            # PREPROCESS
            frame = resize_image(frame, (width, height))

            # flip the image vertically for later selfie view display
            # recolor feed from BGR to RGB so that the model will have good performance
            # frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # to improve performance, mark the image as not writeable to
            # pass by reference instead of making a copy
            frame.flags.writeable = False


            # detect location
            locations = holistic.process(frame) # store all different kinds of landmarks...
            
            # PROCESS
            frame.flags.writeable = True 

            mouth_landmarks = []
            # check if mouth is detected in the frame
            if locations.face_landmarks:
                
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[61].x * width, locations.face_landmarks.landmark[61].y * height]))
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[291].x * width, locations.face_landmarks.landmark[291].y * height]))
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[0].x * width, locations.face_landmarks.landmark[0].y * height]))
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[17].x * width, locations.face_landmarks.landmark[17].y * height]))

                ## calculate the mouth area
                mouth_width = int(cal_euclidean_distance(mouth_landmarks[0], mouth_landmarks[1]))
                #  int(np.sqrt(np.power(locations.face_landmarks.landmark[61].x - locations.face_landmarks.landmark[291].x, 2) + \
                #                     np.power(locations.face_landmarks.landmark[61].y - locations.face_landmarks.landmark[291].y, 2))*width)
                mouth_height = int(cal_euclidean_distance(mouth_landmarks[2], mouth_landmarks[3]))
                # int(np.sqrt(np.power(locations.face_landmarks.landmark[0].x - locations.face_landmarks.landmark[17].x, 2) +
                #                     np.power(locations.face_landmarks.landmark[0].y - locations.face_landmarks.landmark[17].y, 2))*height)

                ## check if the mouth is opened 
                mouth_landmarks.append(np.array([locations.face_landmarks.landmark[14].x * width, locations.face_landmarks.landmark[14].y * height]))
                vector_1 = mouth_landmarks[4] - mouth_landmarks[1]
                vector_2 = mouth_landmarks[4] - mouth_landmarks[0]
                angle = np.arccos(np.dot(vector_1 / np.linalg.norm(vector_1), \
                                                    vector_2 / np.linalg.norm(vector_2))) * PI
                
                # mouth_landmarks.append(np.array([locations.face_landmarks.landmark[82].x * width, locations.face_landmarks.landmark[82].y * height]))
                # mouth_landmarks.append(np.array([locations.face_landmarks.landmark[87].x * width, locations.face_landmarks.landmark[87].y * height]))
                # mouth_landmarks.append(np.array([locations.face_landmarks.landmark[312].x * width, locations.face_landmarks.landmark[312].y * height]))
                # mouth_landmarks.append(np.array([locations.face_landmarks.landmark[317].x * width, locations.face_landmarks.landmark[317].y * height]))
                
                if angle < OPENED_THRESHOLD: 
                    # opacity_level = 0
                    # ROI = dict()
                    # segment selfie image => get mask
                    # mask = (selfie_segmentation.process(frame).segmentation_mask * 255).astype('uint8')
                    mask = np.zeros((height, width), dtype='uint8')
                    # print(f'mask :{mask.shape}')
                    # print(frame.shape)
                    mask[locations.segmentation_mask > tightness] = 255
                    
                    # cv2.i('mask', mask)
                    # print(np.sum(mask>0))
                    # mask[mask <= tightness] = 0
                    face = cv2.bitwise_and(frame, frame, mask=mask)
                    # mask[mask <= 25] = 0
                    # mask[mask > 25] = 255
                    # cv2.imwrite('./proof/original.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    # cv2.imwrite('./proof/mask.png', mask)
                    # cv2.imshow('mask', mask)
                    # Find Canny edges
                    edged = cv2.Canny(mask, 30, 200)
                    # cv2.imwrite('./proof/edges.png', edged)
                    # cv2.imwrite('./proof/segmented_face.png', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    face = cv2.drawContours(face, contours, -1, (255, 255, 255), 10)
                    # cv2.imwrite('./proof/edge_face.png', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                    # blend background
                    # frame[mask == 0] = frame[mask == 0] * (1-alpha) + effect[mask == 0] * (alpha)
                    bg_img = frame.copy()
                    # Create the overlay
                    frame[mask == 0] = cv2.addWeighted(bg_img, 1-alpha, effect[effect_count%no_eff_frames], alpha, 0.0)[mask == 0]
                    # cv2.imwrite('./proof/before_blurr.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    # cv2.imshow('âda',dst)
                    # frame[mask == 0] = frame[mask == 0] * (1-alpha) + effect[effect_count%no_eff_frames][mask == 0] * (alpha)
                    blurred_bg = cv2.GaussianBlur(frame, (21, 21), 0)
                    frame[face == (255, 255, 255)] = blurred_bg[face == (255, 255, 255)]

                    # cv2.imwrite('./proof/after_blurr.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    # frame[mask > 0] = face[mask > 0] 

                    gif_frame = gif_frames[gif_count%no_frames]
                    ### resize the gif frame to fit the mouth area
                    ## resize
                    gif_frame = resize_image(gif_frame, (mouth_width, mouth_height))
                    
                    ### translate the gif frame
                    ## calculate the coords to translate to 
                    tx = mouth_landmarks[0][0]#locations.face_landmarks.landmark[61].x * width
                    ty = mouth_landmarks[2][1]#locations.face_landmarks.landmark[0].y * height
                    ## init a translation matrix
                    translation_matrix = np.array([
                        [1, 0, tx],
                        [0, 1, ty]
                    ], dtype=np.float32)
                    ## translate to the calculated area
                    gif_frame = cv2.warpAffine(src=gif_frame, M=translation_matrix, \
                                                dsize=(width, height), \
                                                borderMode=cv2.BORDER_REPLICATE) 
                    # ROI['mouth_width'] = mouth_width
                    # ROI['mouth_height'] = mouth_height
                    # ROI['mouth_ratio'] = mouth_ratio
                    # ROI['translation_matrix'] = translation_matrix
                    # cv2.imshow('translated' , gif_frame)
                    center_x = int(abs(mouth_landmarks[0][0] + mouth_landmarks[1][0]) / 2)
                    center_y = int(abs(mouth_landmarks[2][1] + mouth_landmarks[3][1]) / 2)
                    
                    vector_1 = [int((- mouth_landmarks[0][0] + mouth_landmarks[1][0])), \
                                    int((- mouth_landmarks[0][1] + mouth_landmarks[1][1]))]

                    # center_x = int(abs(locations.face_landmarks.landmark[61].x + locations.face_landmarks.landmark[291].x)*width / 2)
                    # center_y = int(abs(locations.face_landmarks.landmark[0].y + locations.face_landmarks.landmark[17].y)*height / 2)
                    
                    # vector_1 = [int((- locations.face_landmarks.landmark[61].x + locations.face_landmarks.landmark[291].x)*width), \
                    #                 int((- locations.face_landmarks.landmark[61].y + locations.face_landmarks.landmark[291].y)*height)]
                    vector_2 = [1, 0]
                    angle = np.arccos(np.dot(vector_1 / np.linalg.norm(vector_1), \
                                                vector_2 / np.linalg.norm(vector_2)))*57.2958
                    if (mouth_landmarks[0][1] < mouth_landmarks[1][1]): angle *= -1
                    # if (locations.face_landmarks.landmark[61].y < locations.face_landmarks.landmark[291].y): angle *= -1

                    rotate_matrix = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=angle, scale=1.5)
                    gif_frame = cv2.warpAffine(src=gif_frame, M=rotate_matrix, \
                                                dsize=(width, height), \
                                                borderMode=cv2.BORDER_REPLICATE)

                    # ROI['rotate_matrix'] = rotate_matrix

                    # add animation to the frame
                    # x, y = mouth_landmarks[0]
                    # create the translation matrix using tx and ty, it is a NumPy array 
                    # print(gif_frame.shape)
                    # print(gif_frame.shape)
                    # print(frame.shape)
                    gray_gif = cv2.cvtColor(gif_frame, cv2.COLOR_RGB2GRAY)
                    # cv2.imshow('gray gif', gray_gif)
                    gray_gif[gray_gif < 10] = 255
                    # cv2.imshow('gray', gray_gif)
                    # cv2.imshow('rotated gray gif', gray_gif)
                    frame[gray_gif < 255] = cv2.addWeighted(frame, 0, gif_frame, 1, 0.0)[gray_gif < 255]
                    # Resume the music
                    # mixer.music.play()# Resume the music
                    mixer.music.unpause()
                    # break
                    # frame[gray_gif < 255] = gif_frame[gray_gif < 255] 
                    # for dim in range(3):
                    #     gif_frame[gif_frame[:, :, dim] == 0] = 255
                    #     # frame = gif_frame
                    #     frame[:, :, dim][gif_frame[:, :, dim] < 255] = gif_frame[:,:, dim][gif_frame[:, :, dim] < 255]
                    # break
                else:
                    mixer.music.pause()
                    mixer.music.set_pos(0)
                # elif opacity_level < len(opacity):
                    
                #     # print(ROI)
                #     # print(f'opac lvl {opacity_level}')
                    
                #     gif_frame = resize_image(gif_frames[gif_count%no_frames], (ROI['mouth_width'], ROI['mouth_height']))
                #     ## translate to the calculated area
                #     gif_frame = cv2.warpAffine(src=gif_frame, M=ROI['translation_matrix'], \
                #                                 dsize=(width, height), \
                #                                 borderMode=cv2.BORDER_TRANSPARENT) 
                #     gif_frame = cv2.warpAffine(src=gif_frame, M=ROI['rotate_matrix'], \
                #                                     dsize=(width, height), \
                #                                     borderMode=cv2.BORDER_TRANSPARENT)

                #     for dim in range(3):
                #         gif_frame[gif_frame[:, :, dim] == 0] = 255
                #         frame[:, :, dim][gif_frame[:, :, dim] < 255] = frame[:,:, dim][gif_frame[:, :, dim] < 255]*(1-opacity[opacity_level])+ opacity[opacity_level] * gif_frame[:,:, dim][gif_frame[:, :, dim] < 255]
                #     opacity_level += 1
            
            # res.append(frame)
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
            # if cv2.waitKey(1) & 0xFF == ord('1'): # press 1 to turn on filter
            #     count = 1
                # turn_on_filter = 1
            # move to next frame of the gif
            # if count < no_frames*speed*2 - 2:
            gif_count += 1
            # if effect_count < 
            effect_count += 1
            # else: 
            #     count = 0

    # After the loop release the cap object
    cap.release()
    
    # Destroy all the windows
    cv2.destroyAllWindows()
    # audio.close_player()
    # imageio.mimsave('./output_demo/manyell.gif', res, fps=10)
    # print('Saving mouth dataset..........')
    # print(dataset)



if __name__ == '__main__':
    mouth_gif()