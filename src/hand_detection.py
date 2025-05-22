import cv2
import mediapipe as mp
import numpy as np
import sys
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_finger_extended(finger_tip, finger_pip, finger_mcp, min_distance=0.1):
    # More strict check for finger extension
    return (finger_tip.y < finger_pip.y < finger_mcp.y and 
            abs(finger_tip.y - finger_mcp.y) > min_distance)

def detect_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Strict check for thumb up position
    thumb_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y and 
                abs(thumb_tip.x - thumb_ip.x) < 0.1)  # Thumb should be vertical
    
    # Other fingers should be clearly folded
    others_folded = all([
        tip.y > wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
    ])
    
    return thumb_up and others_folded

def detect_victory(hand_landmarks):
    # Get all necessary landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Check for V shape
    index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
    
    # Others must be clearly folded
    others_folded = all([
        tip.y > wrist.y for tip in [ring_tip, pinky_tip]
    ])
    
    # V shape check
    v_shape = abs(index_tip.x - middle_tip.x) > 0.1  # Fingers must be spread apart
    
    return index_extended and middle_extended and others_folded and v_shape

def detect_open_palm(hand_landmarks):
    # Get all landmarks
    landmarks = {
        'index': (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                 hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                 hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]),
        'middle': (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                  hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                  hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]),
        'ring': (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]),
        'pinky': (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                 hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
                 hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP])
    }
    
    # Check if all fingers are extended
    fingers_extended = all(
        is_finger_extended(tip, pip, mcp)
        for tip, pip, mcp in landmarks.values()
    )
    
    # Check if fingers are spread apart
    finger_tips = [landmarks[finger][0] for finger in ['index', 'middle', 'ring', 'pinky']]
    spread_apart = all(
        abs(finger_tips[i].x - finger_tips[i+1].x) > 0.04
        for i in range(len(finger_tips)-1)
    )
    
    return fingers_extended and spread_apart

def main():
    cap = cv2.VideoCapture(0)
    window_name = 'Live Hand Object Detection'
    cv2.namedWindow(window_name)

    # Dictionary to store gesture detection times
    gesture_times = {
        'thumbs_up': 0,
        'victory': 0,
        'open_palm': 0
    }
    
    # Dictionary for gesture messages
    gesture_messages = {
        'thumbs_up': 'Okay!!',
        'victory': 'Peace!',
        'open_palm': 'Hello!'
    }

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_time = time.time()

            # Clear old detections (more than 3 seconds old)
            for gesture in gesture_times:
                if current_time - gesture_times[gesture] > 2:  # Reduced to 2 seconds
                    gesture_times[gesture] = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Detect gestures
                    if detect_thumbs_up(hand_landmarks):
                        gesture_times['thumbs_up'] = current_time
                    elif detect_victory(hand_landmarks):
                        gesture_times['victory'] = current_time
                    elif detect_open_palm(hand_landmarks):
                        gesture_times['open_palm'] = current_time

            # Display messages for gestures detected within last 2 seconds
            y_position = 50
            for gesture, last_time in gesture_times.items():
                if current_time - last_time < 2:  # Reduced to 2 seconds
                    message = gesture_messages[gesture]
                    cv2.putText(frame, message, (50, y_position), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, 
                              cv2.LINE_AA)
                    y_position += 60

            # Add gesture guide
            guide_text = "Gestures: Thumbs Up, Peace Sign, Open Palm"
            cv2.putText(frame, guide_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, 
                       cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()