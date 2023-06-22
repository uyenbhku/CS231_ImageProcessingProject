# import dependencies
import cv2  # connect webcam, process images
import mediapipe as mp  # holistic API
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

    elif asset_extension in ('gif', 'mp4', 'webm', 'mkv', 'mpeg'):
        asset_gif = cv2.VideoCapture(path)
        while asset_gif.isOpened():
            is_success, f = asset_gif.read()
            if not is_success: 
                break
            # print(f.shape)
            if window_size:
                f = resize_image(f, window_size)
            asset.append(f)
            
        asset = np.array(asset)

    else:
        print(f'unidentified file format, path = {path}')
        # early stop the programme
        return None

    return asset



def cal_euclidean_distance(lm1, lm2):
    return np.sqrt(np.power(lm1[0] - lm2[0], 2) + np.power(lm1[1] - lm2[1], 2))


def main():

    width, height = 1280, 720
    # setting mouth filter
    print('Reading mouth effect.................')
    # read effect
    mouth_effect_path = '.\\assets\\background\\neon_hearts.mp4'
    mouth_effect = read_asset(mouth_effect_path, (width, height))
    if mouth_effect is None: 
        print('Cannot read mouth effect')
        sys.exit(0)
    mouth_effect_count = 0
    no_mouth_eff_frames = mouth_effect.shape[0]
    # define effect opacity
    mouth_alpha = 0.25

    print('Reading mouth gif.................')
    # read gif frames
    gif_path = '.\\assets\\gifs\\ezgif.com-crop.gif'
    mouth_gif_frames = read_asset(gif_path)
    if mouth_gif_frames is None: 
        print('Cannot read mouth gif')
        sys.exit(0)
    
    print('Reading mouth audio..........')
    mouth_audio_path = './audio/ily_voice.mp3'

    no_mouth_frames = mouth_gif_frames.shape[0]
    mouth_gif_count = 0


    ## setting pose filter
    # Load effect angel_wings
    wings_path = './assets/gifs/output_angel_wings.mp4'
    print('Reading wing............')
    # Load effect angel_wings
    angel_wings_animation = cv2.VideoCapture(wings_path)
    angel_wings_frame_count = 0

    # Load effect heart_light
    print('Reading love light ............')
    love_light_path = './assets/background/resized_heart_light.mp4'
    love_light_animation = cv2.VideoCapture(love_light_path)
    love_light_frame_count = 0

    # Sound
    pose_audio_path = './audio/magic_song.mp3'

    # GET REALTIME WEBCAM FEED
    print('Getting webcam feed.................')
    ## define a video capture object, 0 is the webcam
    ## by default, each frame has size (480x640) (height x width)
    # start, end = 0, 0 # helper variables to calculate FPS
    demo_path = '.\\assets\\demo\\womanyell.mp4'
    cap = cv2.VideoCapture(0)
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
    
    choice = 0

    hand_effect = cv2.VideoCapture('./assets/gifs/eff.gif')
    hand_effect_count = 0
    morning = cv2.VideoCapture('./assets/background/morning.gif')
    magic = cv2.VideoCapture('./assets/background/magic.gif')
    pink = cv2.VideoCapture('./assets/background/pink.gif')
    magic_count = 0 
    hand_effect_count = 0
    morning_count = 0
    hand_effect_count = 0
    pink_count = 0

    print('Opening webcam feed........... Press ESC to stop')
    with mp_holistic.Holistic( \
                                # model_complexity=2,
                                enable_segmentation=True,
                                min_detection_confidence=0.5, \
                                min_tracking_confidence=0.5) as holistic:
        # mask = None
        while cap.isOpened():
            # start = time.time()
            # Capture the video frame by frame
            success, frame = cap.read()
            
            if not success:
                print('Cannot receive frame from camera')
                break
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
            
            # recolor feed from RGB to BGR so it can be displayed by openCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            #### mouth filter
            if (choice == 1): 
                mouth_landmarks = []
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
                    # if mouth is opened
                    if angle < OPENED_THRESHOLD: 
                        
                        # get the mask of the background
                        mask = np.zeros((height, width), dtype='uint8')
                        mask[locations.segmentation_mask > tightness] = 255
                        face = cv2.bitwise_and(frame, frame, mask=mask)

                        # Find Canny edges of the mask
                        edged = cv2.Canny(mask, 30, 200)
                        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        face = cv2.drawContours(face, contours, -1, (255, 255, 255), 10)

                        # blend background
                        bg_img = frame.copy()
                        # Create the overlay
                        frame[mask == 0] = cv2.addWeighted(bg_img, 1-mouth_alpha, mouth_effect[mouth_effect_count%no_mouth_eff_frames], mouth_alpha, 0.0)[mask == 0]
                        
                        blurred_bg = cv2.GaussianBlur(frame, (21, 21), 0)
                        frame[face == (255, 255, 255)] = blurred_bg[face == (255, 255, 255)]

                        gif_frame = mouth_gif_frames[mouth_gif_count%no_mouth_frames]
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
                        # rotation
                        center_x = int(abs(mouth_landmarks[0][0] + mouth_landmarks[1][0]) / 2)
                        center_y = int(abs(mouth_landmarks[2][1] + mouth_landmarks[3][1]) / 2)
                        
                        vector_1 = [int((- mouth_landmarks[0][0] + mouth_landmarks[1][0])), \
                                        int((- mouth_landmarks[0][1] + mouth_landmarks[1][1]))]

                        vector_2 = [1, 0]
                        # calculate angle to rotate and convert to degree
                        angle = np.arccos(np.dot(vector_1 / np.linalg.norm(vector_1), \
                                                    vector_2 / np.linalg.norm(vector_2)))*57.2958
                        if (mouth_landmarks[0][1] < mouth_landmarks[1][1]): angle *= -1

                        # create rotate matrix and rotate
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

            #### pose filter
            if (choice == 2):
                ### HIEU UNG DOI CANH THIEN THAN ###
                #----------------------------------------------------------------------------------------#
                if locations.pose_landmarks:
                    poseList = []
                    myPose = locations.pose_landmarks.landmark
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
                        frame[locations.segmentation_mask > 0.5] = cv2.bitwise_not(frame)[locations.segmentation_mask > 0.5]
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
                        
                            frame[locations.segmentation_mask[:,:] > 0.5] = frame_temp[locations.segmentation_mask[:,:] > 0.5]

                            frame[locations.segmentation_mask[:,:] > 0.5] = 0.6*frame[locations.segmentation_mask[:,:] > 0.5] + \
                                0.4*love_light_frame[locations.segmentation_mask[:,:] > 0.5]
                        elif (poseList[1][1] <= poseList[2][1]):
                            frame[locations.segmentation_mask[:,:] > 0.5] = 0.6*frame[locations.segmentation_mask[:,:] > 0.5] + \
                                0.4*love_light_frame[locations.segmentation_mask[:,:] > 0.5]
                            
                            frame[mask_wing[:,:] > 10] = angel_wings_frame[mask_wing[:,:] > 10]
                    else:
                        mixer.music.stop()
                        count_sound = 1

            #### hand gif 
            if (choice == 3):
                # frame = cv2.rotate(frame, cv2.ROTATE_180)
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
                beta = 0.05
                gamma = 0.6

                if results.left_hand_landmarks and results.right_hand_landmarks:
                    # hand_effect_count = 0
                    # hand_effect_count = 0
                    if abs(results.left_hand_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[6].x) < 0.05 \
                        and abs(results.left_hand_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[6].y) < 0.05 \
                        and abs(results.right_hand_landmarks.landmark[3].x - results.right_hand_landmarks.landmark[6].x) < 0.05 \
                        and abs(results.right_hand_landmarks.landmark[3].y - results.right_hand_landmarks.landmark[6].y) < 0.05:
                    
                        okay, bg = hand_effect.read()
                        ok, eff = pink.read()

                        hand_effect_count+=1
                        if hand_effect_count == hand_effect.get(cv2.CAP_PROP_FRAME_COUNT):
                            hand_effect.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            hand_effect_count = 0
                        pink_count+=1
                        if pink_count == pink.get(cv2.CAP_PROP_FRAME_COUNT):
                            pink.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            pink_count = 0

                        ratio = float(hand_effect_count)/hand_effect.get(cv2.CAP_PROP_FRAME_COUNT)
                        bg = cv2.resize(bg, (int(bg.shape[1]*(1+ratio*3)), int(bg.shape[0]*(1+ratio*3))))
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
                        translated_bg_2 = cv2.warpAffine(src=cv2.flip(bg, 1), M=translation_matrix_2, dsize=(fr_h, fr_w))
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
                        frame[gif_1<255] = cv2.addWeighted(frame, alpha + ratio, translated_bg_1, 1 - alpha - ratio, 0)[gif_1<255]
                        gif_2[gif_2==0] = 255
                        frame[gif_2<255] = cv2.addWeighted(frame, alpha + ratio, translated_bg_2, 1 - alpha - ratio, 0)[gif_2<255]

                        # mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        # mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                elif results.left_hand_landmarks and results.right_hand_landmarks is None:
                    # hand_effect_count = 0
                    # hand_effect_count = 0
                    if abs(results.left_hand_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[6].x) < 0.05 and abs(results.left_hand_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[6].y) < 0.05:
                        okay, bg = hand_effect.read()

                        _, eff = morning.read()

                        hand_effect_count+=1
                        if hand_effect_count == hand_effect.get(cv2.CAP_PROP_FRAME_COUNT):
                            hand_effect.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            hand_effect_count = 0
                        morning_count+=1
                        if morning_count == morning.get(cv2.CAP_PROP_FRAME_COUNT):
                            morning.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            morning_count = 0
                        
                        ratio = float(hand_effect_count)/hand_effect.get(cv2.CAP_PROP_FRAME_COUNT)
                        bg = cv2.resize(bg, (int(bg.shape[1]*(1+ratio*3)), int(bg.shape[0]*(1+ratio*3))))
                        w, h, c = bg.shape
                        e_w, e_h, e_c = eff.shape
                        eff = cv2.resize(eff, (fr_h, fr_w))
                        x = int(results.left_hand_landmarks.landmark[3].x *fr_h)
                        y = int(results.left_hand_landmarks.landmark[3].y *fr_w)

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
                        frame[gif<255] = cv2.addWeighted(frame, alpha + ratio, translated_bg, 1 - alpha - ratio, 0)[gif<255]
                            
                    # mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
                elif results.right_hand_landmarks and results.left_hand_landmarks is None:
                    # hand_effect_count = 0
                    # hand_effect_count = 0
                    if abs(results.right_hand_landmarks.landmark[3].x - results.right_hand_landmarks.landmark[6].x) < 0.05 and abs(results.right_hand_landmarks.landmark[3].y - results.right_hand_landmarks.landmark[6].y) < 0.05:
                        okay, bg = hand_effect.read()

                        _, eff = magic.read()
                        hand_effect_count+=1
                        if hand_effect_count == hand_effect.get(cv2.CAP_PROP_FRAME_COUNT):
                            hand_effect.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            hand_effect_count = 0
                        magic_count+=1
                        if magic_count == magic.get(cv2.CAP_PROP_FRAME_COUNT):
                            magic.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            magic_count = 0
                        ratio = float(hand_effect_count)/hand_effect.get(cv2.CAP_PROP_FRAME_COUNT)
                        bg = cv2.resize(bg, (int(bg.shape[1]*(1+ratio*3)), int(bg.shape[0]*(1+ratio*3))))
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
                        edges = cv2.Canny(mask,100,200)
                        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        blend = frame[int(y-w/2):int(y+w/2), int(x-h/2):int(x+h/2), :]
                        frame[mask != 0] = frame[mask != 0] * gamma + eff[mask != 0] * (1 - gamma)
                        blur = cv2.GaussianBlur(frame, (21, 21), 0)
                        cv2.drawContours(mask, contours, -1, (100), 10)
                        frame[mask==100] = blur[mask==100]
                        gif[gif==0] = 255
                        frame[gif<255] = cv2.addWeighted(frame, alpha + ratio, translated_bg, 1 - alpha - ratio, 0)[gif<255]


            # calculate how long this code takes to process a frame on a CPU
            # end = time.time()  
            # fps = 1/(end - start)
            # # display FPS on the frame
            # cv2.putText(frame, str(f'FPS: {int(fps)}'), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3)
            
            # Display the resulting frame
            cv2.imshow('Webcam Feed', frame)

            # choose filter
            key = cv2.waitKey(1)
            if key == 27: # press ESC to terminate 
                break
            if key == ord('1'): # press 1 to turn on mouth filter
                choice = 1
                #Instantiate mixer
                mixer.init()
                # load audio
                mixer.music.load(mouth_audio_path)
                # play the music 
                mixer.music.play(-1)
                # pause 
                mixer.music.pause()
            elif key == ord('2'): # press 2 to turn on pose filter
                choice = 2
                mixer.init()
                mixer.music.load(pose_audio_path)
            elif key == ord('3'): # press 3 to turn on hand filter
                choice = 3

    # After the loop release the cap object
    cap.release()
    
    # Destroy all the windows
    cv2.destroyAllWindows()

    


if __name__ == '__main__':
    main()