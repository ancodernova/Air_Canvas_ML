# Air_Canvas_ML

### Interactive Hand-Gesture Controlled Drawing Application

In this project, we developed an interactive drawing application that uses hand gestures to control drawing on a virtual canvas. Leveraging OpenCV for video processing and MediaPipe for hand tracking, this application detects and interprets hand movements to draw, select colors, change brush sizes, and erase drawings in real-time. Below is a summary of the workflow and key functionalities:

### Workflow:
1. **Initialization**:
   - Initialize deques for storing points of different colors.
   - Define brush sizes, eraser sizes, and set the default size.
   - Create the paint window and a color selection panel.

2. **Hand Detection Setup**:
   - Use MediaPipe's Hand solution to detect and track hand landmarks.
   - Initialize the webcam to capture real-time video frames.

3. **Main Loop**:
   - Capture frames from the webcam, flip them for a mirror effect, and convert them to RGB for processing.
   - Overlay the color selection panel and display brush/eraser size options on the paint window.

4. **Hand Gesture Processing**:
   - Detect hand landmarks using MediaPipe.
   - Identify the positions of the forefinger and thumb to determine gestures.
   - Append detected points to the respective color deque based on the selected color.

5. **Drawing and Interaction**:
   - Draw lines on both the real-time frame and the virtual canvas based on hand movements.
   - Implement color selection and brush/eraser size adjustments through specific gesture positions.
   - Clear the canvas when the "Clear" button is activated through a gesture.

6. **Undo Functionality**:
   - Implement an undo feature using a stack to track the drawing actions, allowing users to revert their last action by pressing the 'u' key.

7. **Display and Exit**:
   - Continuously display the updated frames and paint window.
   - Exit the application gracefully when the 'q' key is pressed.

### Key Features:
- **Real-time Hand Tracking**: Utilizes MediaPipe for accurate hand landmark detection.
- **Dynamic Drawing**: Supports drawing with multiple colors, adjustable brush sizes, and an eraser tool.
- **Intuitive UI**: Color selection panel and size options for an enhanced user experience.
- **Undo Functionality**: Allows users to revert their last action for improved control.

This project showcases the integration of computer vision and machine learning techniques to create an interactive and user-friendly drawing application. The complete code can be found. 

