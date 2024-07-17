import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize color points deque and indices
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
kpoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
black_index = 0

# Brush sizes and default size
brush_sizes = [3, 5, 7, 9, 11]
brush_size = brush_sizes[0]
eraser_sizes = [20, 30, 40]
eraser_size = eraser_sizes[0]

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Define colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0)]
color_names = ["BLUE", "GREEN", "RED", "YELLOW", "BLACK"]
colorIndex = 0

# Set up the paint window
paintWindow = np.zeros((471, 1036, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Initialize color selection panel
color_panel = np.zeros((471, 100, 3), dtype=np.uint8)

# Populate color selection panel
panel_color_step = int(471 / len(colors))
for i, color in enumerate(colors):
    color_panel[i * panel_color_step:(i + 1) * panel_color_step, :, :] = color

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
undo_stack = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Overlay color_panel on paintWindow
    paintWindow[:, :640, :] = 255  # Background for drawing canvas
    paintWindow[:, 640:740, :] = color_panel  # Color selection panel

    # Display brush and eraser size options
    cv2.putText(paintWindow, "BRUSH SIZE: {}".format(brush_size), (750, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "ERASER SIZE: {}".format(eraser_size), (750, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Process the results
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        if (thumb[1] - center[1] < 30):
            if colorIndex < len(colors):  # Only append for colors, not eraser
                undo_stack.append((colorIndex, center))
                if colorIndex == 0:
                    bpoints.append(deque(maxlen=512))
                    blue_index += 1
                elif colorIndex == 1:
                    gpoints.append(deque(maxlen=512))
                    green_index += 1
                elif colorIndex == 2:
                    rpoints.append(deque(maxlen=512))
                    red_index += 1
                elif colorIndex == 3:
                    ypoints.append(deque(maxlen=512))
                    yellow_index += 1
                elif colorIndex == 4:
                    kpoints.append(deque(maxlen=512))
                    black_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                kpoints = [deque(maxlen=512)]
                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
                black_index = 0
                undo_stack = []  # Clear undo stack
                paintWindow[67:, :640, :] = 255  # Clear drawing canvas
            elif 640 <= center[0] <= 740:  # Color selection panel
                colorIndex = int(center[1] // panel_color_step)
            elif 750 <= center[0] <= 780:  # Brush size selection
                brush_size = brush_sizes[int((center[1] - 65) // 50)]
            elif 800 <= center[0] <= 830:  # Eraser size selection
                eraser_size = eraser_sizes[int((center[1] - 110) // 50)]
        else:
            if colorIndex < len(colors):
                points = [bpoints, gpoints, rpoints, ypoints, kpoints]
                indices = [blue_index, green_index, red_index, yellow_index, black_index]
                if colorIndex < len(points):
                    points[colorIndex][indices[colorIndex]].appendleft(center)

                # Draw lines on canvas and frame
                for i in range(len(points)):
                    for j in range(len(points[i])):
                        for k in range(1, len(points[i][j])):
                            if points[i][j][k - 1] is not None and points[i][j][k] is not None:
                                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], brush_size)
                                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], brush_size)

            elif colorIndex == len(colors):  # Eraser
                cv2.circle(paintWindow, center, eraser_size, (255, 255, 255), -1)

    # Handle undo functionality
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u'):
        if undo_stack:
            colorIndex, center = undo_stack.pop()
            if colorIndex < len(colors):
                if colorIndex == 0 and blue_index > 0:
                    bpoints.pop()
                    blue_index -= 1
                elif colorIndex == 1 and green_index > 0:
                    gpoints.pop()
                    green_index -= 1
                elif colorIndex == 2 and red_index > 0:
                    rpoints.pop()
                    red_index -= 1
                elif colorIndex == 3 and yellow_index > 0:
                    ypoints.pop()
                    yellow_index -= 1
                elif colorIndex == 4 and black_index > 0:
                    kpoints.pop()
                    black_index -= 1
            elif colorIndex == len(colors):  # Eraser
                cv2.circle(paintWindow, center, eraser_size, (255, 255, 255), -1)

    # Exit on 'q' key press
    if key == ord('q'):
        break

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

cap.release()
cv2.destroyAllWindows()
