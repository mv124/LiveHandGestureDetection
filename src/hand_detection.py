import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for natural interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR image to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the wrist (landmark 0) and middle finger MCP (landmark 9)
                x_list = [lm.x * w for lm in hand_landmarks.landmark]
                y_list = [lm.y * h for lm in hand_landmarks.landmark]
                center_x = int(np.mean(x_list))
                center_y = int(np.mean(y_list))
                # Estimate radius as max distance from center to any landmark
                radius = int(max(np.sqrt((np.array(x_list)-center_x)**2 + (np.array(y_list)-center_y)**2)))
                # Draw the circle
                cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
                # Optionally, draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Live Hand Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()