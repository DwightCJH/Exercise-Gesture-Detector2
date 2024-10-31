import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.6)

# Set up video capture
cap = cv2.VideoCapture(0)

# Timer settings
COUNTDOWN_TIME = 20  # 20 seconds countdown
timer = COUNTDOWN_TIME
paused = True  # Start with the timer paused
start_time = None  # Keeps track of when the timer starts or resumes
elapsed_paused_time = 0  # Total paused time
pause_start_time = None  # Track when the timer was paused

# Gesture-based exercise status
exercise_status = "Exercise Paused"
status_color = (0, 0, 255)  # Red for paused
thumbs_up_time = 0
thumbs_down_time = 0
switch_time_threshold = 0.3  # seconds

def update_timer():
    """Update the countdown timer."""
    global timer, start_time, elapsed_paused_time

    if not paused and start_time:  # Only update if the timer is running
        current_time = time.time()
        elapsed_time = current_time - start_time - elapsed_paused_time
        timer = max(0, COUNTDOWN_TIME - int(elapsed_time))

def toggle_pause():
    """Toggle the pause state and update exercise status."""
    global paused, start_time, elapsed_paused_time, pause_start_time, exercise_status, status_color

    if paused:
        # Resuming the timer
        paused = False
        exercise_status = "Exercise in Progress"
        status_color = (0, 255, 0)  # Green
        if start_time is None:
            start_time = time.time()
        else:
            # Adjust the total paused time
            elapsed_paused_time += time.time() - pause_start_time
    else:
        # Pausing the timer
        paused = True
        exercise_status = "Exercise Paused"
        status_color = (0, 0, 255)  # Red
        pause_start_time = time.time()

def reset_timer():
    """Reset the timer to the initial countdown time."""
    global timer, start_time, elapsed_paused_time, pause_start_time
    timer = COUNTDOWN_TIME
    start_time = None
    elapsed_paused_time = 0
    pause_start_time = None
    if not paused:
        toggle_pause()

# Set up mouse callback for detecting button clicks
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_inside_pause_button(x, y):
            toggle_pause()
        elif is_inside_reset_button(x, y):
            reset_timer()

# Assign the mouse callback to the OpenCV window
cv2.namedWindow('Exercise Gesture Control')
cv2.setMouseCallback('Exercise Gesture Control', mouse_callback)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        thumbs_up_count = 0
        thumbs_down_count = 0

        for hand_landmarks in results.multi_hand_landmarks:
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect thumbs up and down gestures
            if thumb_tip_y < wrist_y and thumb_tip_y < index_finger_tip_y:
                thumbs_up_count += 1
            elif thumb_tip_y > wrist_y and thumb_tip_y > index_finger_tip_y:
                thumbs_down_count += 1

        # Switch status based on gestures
        if thumbs_up_count == 2:
            thumbs_up_time += 1 / 30  # Assuming 30 FPS
            thumbs_down_time = 0
        elif thumbs_down_count == 2:
            thumbs_down_time += 1 / 30  # Assuming 30 FPS
            thumbs_up_time = 0
        else:
            thumbs_up_time = 0
            thumbs_down_time = 0

        if thumbs_up_time >= switch_time_threshold and paused:
            toggle_pause()
        elif thumbs_down_time >= switch_time_threshold and not paused:
            toggle_pause()

    # Update and display the timer
    update_timer()
    if timer == 0 and not paused:
        toggle_pause()
    timer_text = f"{timer} sec"
    cv2.putText(image, timer_text, (image.shape[1] - 250, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    # Display exercise status
    cv2.putText(image, exercise_status, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Draw the circular pause/resume button at the bottom center
    button_radius = 50
    pause_button_x = image.shape[1] // 2
    pause_button_y = image.shape[0] - button_radius - 20
    button_color = (0, 255, 0) if paused else (0, 0, 255)
    cv2.circle(image, (pause_button_x, pause_button_y), button_radius, button_color, -1)
    cv2.circle(image, (pause_button_x, pause_button_y), button_radius, (255, 255, 255), 2)
    button_text = "Resume" if paused else "Pause"
    text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = pause_button_x - text_size[0] // 2
    text_y = pause_button_y + text_size[1] // 2
    cv2.putText(image, button_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw the reset button in the bottom right corner
    reset_button_width, reset_button_height = 100, 40
    reset_button_x = image.shape[1] - reset_button_width - 20
    reset_button_y = image.shape[0] - reset_button_height - 20
    cv2.rectangle(image, (reset_button_x, reset_button_y), (reset_button_x + reset_button_width, reset_button_y + reset_button_height), (255, 165, 0), -1)
    cv2.rectangle(image, (reset_button_x, reset_button_y), (reset_button_x + reset_button_width, reset_button_y + reset_button_height), (255, 255, 255), 2)
    reset_text = "Reset"
    reset_text_size = cv2.getTextSize(reset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    reset_text_x = reset_button_x + (reset_button_width - reset_text_size[0]) // 2
    reset_text_y = reset_button_y + (reset_button_height + reset_text_size[1]) // 2
    cv2.putText(image, reset_text, (reset_text_x, reset_text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Update the is_inside_button functions
    def is_inside_pause_button(x, y):
        return np.sqrt((x - pause_button_x)**2 + (y - pause_button_y)**2) <= button_radius

    def is_inside_reset_button(x, y):
        return reset_button_x <= x <= reset_button_x + reset_button_width and reset_button_y <= y <= reset_button_y + reset_button_height

    # Show the video feed
    cv2.imshow('Exercise Gesture Control', image)

    # Handle quit event
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()