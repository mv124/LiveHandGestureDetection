import cv2
import mediapipe as mp
import numpy as np
import sys
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_finger_state(hand_landmarks, finger_tip_id, finger_pip_id, finger_mcp_id):
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    mcp = hand_landmarks.landmark[finger_mcp_id]
    
    # Calculate distances
    tip_to_pip = abs(tip.y - pip.y)
    pip_to_mcp = abs(pip.y - mcp.y)
    
    # Finger is extended if tip is above pip and pip is above mcp
    return tip.y < pip.y < mcp.y

def detect_thumbs_up(hand_landmarks):
    # Get thumb landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Get other finger tips
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Thumb must be pointing up (more lenient check)
    thumb_up = (thumb_tip.y < thumb_ip.y and  # Only check if tip is above IP
                abs(thumb_tip.x - thumb_ip.x) < 0.2)  # Allow more horizontal movement
    
    # Other fingers should be folded (more lenient check)
    others_folded = all([
        tip.y > wrist.y - 0.1 for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
    ])
    
    return thumb_up and others_folded

def detect_victory(hand_landmarks):
    # Get finger landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Check if index and middle fingers are extended (more lenient)
    index_extended = (index_tip.y < index_pip.y and  # Tip above PIP
                     abs(index_tip.y - index_pip.y) > 0.02)  # Minimum extension
    
    middle_extended = (middle_tip.y < middle_pip.y and  # Tip above PIP
                      abs(middle_tip.y - middle_pip.y) > 0.02)  # Minimum extension
    
    # Check if other fingers are folded (more lenient)
    ring_folded = ring_tip.y > ring_pip.y  # Tip below PIP
    pinky_folded = pinky_tip.y > pinky_pip.y  # Tip below PIP
    
    # Check V shape (more lenient)
    v_shape = (abs(index_tip.x - middle_tip.x) > 0.02 and  # Minimum spread
              abs(index_tip.y - middle_tip.y) < 0.1)  # Fingers should be roughly at same height
    
    # Debug information
    debug_info = {
        'index_extended': index_extended,
        'middle_extended': middle_extended,
        'ring_folded': ring_folded,
        'pinky_folded': pinky_folded,
        'v_shape': v_shape
    }
    
    return (index_extended and middle_extended and 
            ring_folded and pinky_folded and v_shape)

def detect_open_palm(hand_landmarks):
    # Check each finger
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP)
    ]
    
    # All fingers must be extended
    all_extended = all(
        get_finger_state(hand_landmarks, tip, pip, mcp)
        for tip, pip, mcp in fingers
    )
    
    # Get finger tips for spread check
    finger_tips = [hand_landmarks.landmark[tip] for tip, _, _ in fingers]
    
    # Check if fingers are spread apart
    spread_apart = all(
        abs(finger_tips[i].x - finger_tips[i+1].x) > 0.03
        for i in range(len(finger_tips)-1)
    )
    
    return all_extended and spread_apart

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

    # Add debug information
    debug_info = {
        'thumbs_up': False,
        'victory': False,
        'open_palm': False
    }

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_time = time.time()

            # Clear old detections
            for gesture in gesture_times:
                if current_time - gesture_times[gesture] > 1.5:
                    gesture_times[gesture] = 0
                    debug_info[gesture] = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Detect gestures
                    debug_info['thumbs_up'] = detect_thumbs_up(hand_landmarks)
                    debug_info['victory'] = detect_victory(hand_landmarks)
                    debug_info['open_palm'] = detect_open_palm(hand_landmarks)
                    
                    if debug_info['thumbs_up']:
                        gesture_times['thumbs_up'] = current_time
                    elif debug_info['victory']:
                        gesture_times['victory'] = current_time
                    elif debug_info['open_palm']:
                        gesture_times['open_palm'] = current_time

                    # Display landmark positions for debugging
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.putText(frame, str(idx), (x, y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # Display messages for gestures detected within last 1.5 seconds
            y_position = 50
            for gesture, last_time in gesture_times.items():
                if current_time - last_time < 1.5:
                    message = gesture_messages[gesture]
                    cv2.putText(frame, message, (50, y_position), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, 
                              cv2.LINE_AA)
                    y_position += 60

            # Display debug information
            debug_y = h - 100
            for gesture, detected in debug_info.items():
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(frame, f"{gesture}: {'Yes' if detected else 'No'}", 
                          (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                debug_y += 25

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