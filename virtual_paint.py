import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Set brush color, thickness, and background
brush_color = (0, 0, 255)  # Red color by default
brush_thickness = 10
canvas = None

# Start webcam feed
cap = cv2.VideoCapture(0)

# Function to reset the canvas
def reset_canvas(frame):
    return np.ones_like(frame) * 255  # White canvas, based on frame size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural hand movement (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Initialize canvas when first frame is read
    if canvas is None:
        canvas = reset_canvas(frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of the index finger tip (landmark 8)
            index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x, index_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            # Gesture-based functionalities
            # Draw a circle on the canvas at the index finger tip (virtual paint)
            if 0 <= index_x < frame.shape[1] and 0 <= index_y < frame.shape[0]:
                cv2.circle(canvas, (index_x, index_y), brush_thickness, brush_color, -1)

            # Hand Gesture for Clear Screen (making a fist or a swipe)
            if len(results.multi_hand_landmarks) == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                if abs(thumb_tip.x - index_tip.x) < 0.05:  # Detecting fist gesture
                    canvas = reset_canvas(frame)  # Clear the canvas if a fist is detected

    # Combine the frame and canvas to show the drawing on top of the webcam feed
    combined_frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    # Display the combined frame
    cv2.imshow("Virtual Paint - Hand Gesture Drawing", combined_frame)

    # Exit the app when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
