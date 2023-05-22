import cv2
import mediapipe as mp
import mss
from pynput.mouse import Controller, Button
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

monitors = mss.mss().monitors

# NOTE: We're currently arbitrarily choosing the main monitor, but in the future should select whatever has left and top set to 0/ one currently in use
main_monitor = monitors[0]

monitor_width = main_monitor.get('width')
monitor_height = main_monitor.get('height')

# with mss.mss() as sct:
#   for monitors in sct.monitors:
#     print(sct.monitors)

mouse = Controller()

previous_image = None
left_click_on_cooldown = False
right_click_on_cooldown = False

def detect_click(thumb_tip, middle_tip, pinky_tip):
    global left_click_on_cooldown
    global right_click_on_cooldown

    index_thumb_tip_distance = math.hypot(thumb_tip["x"] - middle_tip["x"], thumb_tip["y"] - middle_tip["y"])  
    pinky_thumb_tip_distance = math.hypot(thumb_tip["x"] - pinky_tip["x"], thumb_tip["y"] - pinky_tip["y"])
     
    # Left click (hold & release/click) based on how close the thumb and middle finger tips are 
    if (left_click_on_cooldown == False):
        if (index_thumb_tip_distance < 0.1):
            mouse.press(Button.left)
            left_click_on_cooldown = True
            print("left click hold!")
    else:
        if (index_thumb_tip_distance > 0.3):
            mouse.release(Button.left)
            left_click_on_cooldown = False
            print("left click release!")

    # Right click (hold & release/click) based on how close the thumb and pinky finger tips are 
    if (right_click_on_cooldown == False):
        if (pinky_thumb_tip_distance < 0.1):
            mouse.press(Button.right)
            right_click_on_cooldown = True
            print("right click hold!")
    else:
        if (pinky_thumb_tip_distance > 0.3):
            mouse.release(Button.right)
            right_click_on_cooldown = False
            print("right click release!")
            
# Gesture Recognition Functions
def recognize_gesture(results, image_width, image_height, image):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Check for click

            # handedness = results.multi_handedness[x].label
            # print(handedness)

            # We flip the x positions to account for the horizontal camera flip.
            index_tip_x = (1 - hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)
            index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Tips of all other fingers to use for click detection
            thumb_tip_x = (1 - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x)
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            middle_tip_x = (1 - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x)
            middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
             
            ring_tip_x = (1 - hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x)
            ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y

            pinky_tip_x = (1 - hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x)
            pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

            # print(
            #     f'Index finger tip coordinates: (',
            #     f'{index_tip_x * image_width}, '
            #     f'{index_tip_y * image_height})'
            # )      

            # print(f'Index Tip: {int(index_tip_x * monitor_width)}, {int(index_tip_y * monitor_height)}')
            # print(f'Thumb Tip: {int(thumb_tip_x * monitor_width)}, {int(thumb_tip_y * monitor_height)}')
            # print(f'Middle Tip: {int(middle_tip_x * monitor_width)}, {int(middle_tip_y * monitor_height)}')
            # print(f'Ring Tip: {int(ring_tip_x * monitor_width)}, {int(ring_tip_y * monitor_height)}')
            # print(f'Pinky Tip: {int(pinky_tip_x * monitor_width)}, {int(pinky_tip_y * monitor_height)}')
            # print(" ")

            thumb_tip = {"x": thumb_tip_x, "y": thumb_tip_y}
            middle_tip = {"x": middle_tip_x, "y": middle_tip_y}
            pinky_tip = {"x": pinky_tip_x, "y": pinky_tip_y}

            detect_click(thumb_tip, middle_tip, pinky_tip)

            old_position = list(mouse.position)
            old_position.append('old')
            new_position = [int(index_tip_x * monitor_width), int(index_tip_y * monitor_height), 'new']

            mouse.position = (new_position[0], new_position[1])

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            recognize_gesture(results, image_width, image_height, image)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
