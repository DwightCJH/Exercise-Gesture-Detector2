import cv2
import mediapipe as mp
import numpy as np
import math

class ExerciseTracker:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize tracking variables
        self.rep_count = 0
        self.position_state = "up"
        self.squat_threshold = 120
        self.is_tracking = True
        
        # Add timing variables for gesture detection
        self.thumbs_up_time = 0
        self.thumbs_down_time = 0
        self.switch_time_threshold = 0.3  # seconds
        
        # Add reset button variables
        self.reset_button = {
            'x': 0,  # Will be set in process_frame
            'y': 0,
            'width': 120,
            'height': 50
        }
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                 np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def detect_thumb_gesture(self, hand_landmarks):
        """Detect if thumb is pointing up or down"""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Check for clear thumbs up/down gesture
        if thumb_tip.y < wrist.y and thumb_tip.y < index_tip.y:
            return "up"
        elif thumb_tip.y > wrist.y and thumb_tip.y > index_tip.y:
            return "down"
        return "neutral"
    
    def reset_workout(self):
        """Reset all tracking variables"""
        self.rep_count = 0
        self.position_state = "up"
        self.is_tracking = True
        self.thumbs_up_time = 0
        self.thumbs_down_time = 0
    
    def is_inside_reset_button(self, x, y):
        """Check if coordinates are inside reset button"""
        return (self.reset_button['x'] <= x <= self.reset_button['x'] + self.reset_button['width'] and
                self.reset_button['y'] <= y <= self.reset_button['y'] + self.reset_button['height'])
    
    def draw_reset_button(self, image):
        """Draw reset button on the frame"""
        # Update button position based on frame size
        self.reset_button['x'] = image.shape[1] - self.reset_button['width'] - 20
        self.reset_button['y'] = image.shape[0] - self.reset_button['height'] - 20
        
        # Draw button background
        cv2.rectangle(image,
                     (self.reset_button['x'], self.reset_button['y']),
                     (self.reset_button['x'] + self.reset_button['width'],
                      self.reset_button['y'] + self.reset_button['height']),
                     (255, 165, 0),  # Orange background
                     -1)  # Filled rectangle
        
        # Draw button border
        cv2.rectangle(image,
                     (self.reset_button['x'], self.reset_button['y']),
                     (self.reset_button['x'] + self.reset_button['width'],
                      self.reset_button['y'] + self.reset_button['height']),
                     (255, 255, 255),  # White border
                     2)  # Border thickness
        
        # Add text
        reset_text = "Reset"
        text_size = cv2.getTextSize(reset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = self.reset_button['x'] + (self.reset_button['width'] - text_size[0]) // 2
        text_y = self.reset_button['y'] + (self.reset_button['height'] + text_size[1]) // 2
        
        cv2.putText(image, reset_text,
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,  # Font scale
                   (255, 255, 255),  # White text
                   2)  # Text thickness
        
        return image
    
    def process_gestures(self, frame):
        """Process hand gestures and return control state"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        thumbs_up_count = 0
        thumbs_down_count = 0
        
        if results.multi_hand_landmarks:
            # Process each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Check gestures
                gesture = self.detect_thumb_gesture(hand_landmarks)
                if gesture == "up":
                    thumbs_up_count += 1
                elif gesture == "down":
                    thumbs_down_count += 1
        
        # Update gesture timers
        if thumbs_up_count == 2:
            self.thumbs_up_time += 1/30  # Assuming 30 FPS
            self.thumbs_down_time = 0
        elif thumbs_down_count == 2:
            self.thumbs_down_time += 1/30
            self.thumbs_up_time = 0
        else:
            self.thumbs_up_time = 0
            self.thumbs_down_time = 0
            
        # Only change tracking state on thumbs down or clear thumbs up
        if self.thumbs_down_time >= self.switch_time_threshold:
            self.is_tracking = False
        elif self.thumbs_up_time >= self.switch_time_threshold and not self.is_tracking:
            self.is_tracking = True
                    
        return frame
    
    def check_squat_form(self, landmarks):
        """Analyze squat form and count reps"""
        # Get relevant landmarks
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate knee angle
        angle = self.calculate_angle(hip, knee, ankle)
        
        # Count reps based on angle thresholds
        if self.is_tracking:  # Only count reps when tracking is enabled
            if angle < self.squat_threshold and self.position_state == "up":
                self.position_state = "down"
            elif angle > 160 and self.position_state == "down":
                self.position_state = "up"
                self.rep_count += 1
        
        # Analyze form
        if angle < 90:
            return f"Too low! Reps: {self.rep_count}", "red", angle
        elif angle > 150:
            return f"Stand straight! Reps: {self.rep_count}", "white", angle
        else:
            return f"Good form! Reps: {self.rep_count}", "green", angle
    
    def create_pause_screen(self, frame_shape):
        """Create a white modal popup with pause message"""
        black_screen = np.zeros(frame_shape, dtype=np.uint8)
        height, width = frame_shape[:2]
        
        # Define the size and position of the modal popup
        modal_width = int(width * 0.6)  # 60% of the screen width
        modal_height = int(height * 0.4)  # 40% of the screen height
        modal_x = (width - modal_width) // 2
        modal_y = (height - modal_height) // 2
        
        # Draw white background for the modal popup
        cv2.rectangle(black_screen,
                    (modal_x, modal_y),
                    (modal_x + modal_width, modal_y + modal_height),
                    (255, 255, 255),  # White background
                    -1)  # Filled rectangle
        
        # Draw black border around the modal
        cv2.rectangle(black_screen,
                    (modal_x, modal_y),
                    (modal_x + modal_width, modal_y + modal_height),
                    (0, 0, 0),  # Black border
                    5)  # Border thickness
        
        # Define the text to display
        pause_text = "WORKOUT PAUSED"
        instruction_text = "Show thumbs up to resume"
        
        # Get text sizes
        pause_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
        instruction_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        
        # Calculate center positions
        pause_x = (modal_width - pause_size[0]) // 2 + modal_x
        instruction_x = (modal_width - instruction_size[0]) // 2 + modal_x
        
        pause_y = modal_y + int(modal_height * 0.2)  # 20% of the modal height from the top
        instruction_y = pause_y + int(pause_size[1] * 1.5)  # 1.5x the height of the pause text
        
        # Add pause message
        cv2.putText(black_screen, pause_text,
                    (pause_x, pause_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 
                    (0, 0, 0),  # Black text
                    4)
        
        # Add instruction message
        cv2.putText(black_screen, instruction_text,
                    (instruction_x, instruction_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                    (0, 0, 0),  # Black text
                    3)
        
        # Draw current rep count
        rep_text = f"Current Reps: {self.rep_count}"
        rep_size = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        rep_x = (modal_width - rep_size[0]) // 2 + modal_x
        rep_y = instruction_y + 60  # 60px below the instruction text
        
        cv2.putText(black_screen, rep_text,
                    (rep_x, rep_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 0),  # Black text
                    3)
        
        # Draw reset button on pause screen
        black_screen = self.draw_reset_button(black_screen)
        
        return black_screen
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Process gestures first (this updates is_tracking state)
        frame = self.process_gestures(frame)
        
        # If not tracking, show pause screen
        if not self.is_tracking:
            return self.create_pause_screen(frame.shape)
        
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make pose detection
        results = self.pose.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get frame dimensions
        frame_height, frame_width, _ = image.shape
        
        if results.pose_landmarks:
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                image, 
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Check form
            feedback, color, angle = self.check_squat_form(results.pose_landmarks.landmark)
            
            # Calculate positions for text
            rep_y_position = int(frame_height * 0.1)
            feedback_y_position = int(frame_height * 0.2)
            angle_y_position = int(frame_height * 0.3)
            
            # Display rep count
            cv2.putText(image, f"REPS: {self.rep_count}", 
                       (30, rep_y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, 
                       (255, 255, 255), 
                       4)
            
            # Display feedback
            cv2.putText(image, feedback, 
                       (30, feedback_y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                       (0, 255, 0) if color == "green" else (0, 0, 255), 
                       3)
            
            # Display angle
            cv2.putText(image, f"Knee Angle: {int(angle)}", 
                       (30, angle_y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                       (255, 255, 255), 
                       3)
        
        # Draw reset button
        image = self.draw_reset_button(image)
        
        return image
    
    def start_tracking(self):
        """Start webcam tracking"""
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create named window and set mouse callback
        window_name = 'Exercise Form Tracking'
        cv2.namedWindow(window_name)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.is_inside_reset_button(x, y):
                    self.reset_workout()
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1) 

            # Process frame
            image = self.process_frame(frame)

            # Display
            cv2.imshow(window_name, image)

            # Break on 'q' press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    tracker = ExerciseTracker()
    tracker.start_tracking()
