import cv2
import numpy as np

#Initializes the video capture object 'cap' by opening the video file 'ControlStable.mp4'.
cap = cv2.VideoCapture('ControlStable.mp4')
#cap = cv2.VideoCapture('52.mp4')
#cap = cv2.VideoCapture('ControlStable.mp4')
#cap = cv2.VideoCapture('52.mp4')
#cap = cv2.VideoCapture('ControlStable.mp4')
#cap = cv2.VideoCapture('Etoh5.mp4')
#cap = cv2.VideoCapture('EtOH001.mp4')
#cap = cv2.VideoCapture('C60-0036.mp4')
#cap = cv2.VideoCapture('C60-0006.mp4')
#cap = cv2.VideoCapture('Caffeine0001.mp4')

# Check if video file is opened successfully
# If there's an error opening the video file, it prints an error message and exits the program.
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Background subtractor object is created
# Will be used to separate moving foreground objects from the static background in the video frames
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Initialize tracking information
planaria_tracks = {}
planaria_id_counter = 0
# Speed calculation (in pixels per second)
planaria_speeds = {}
# Starts a loop that iterates through each frame of the video until it reaches the end.
while cap.isOpened():
    # reads a frame from the video capture object cap
    # cap.read() function returns two values: ret, which indicates if a frame was successfully read, and frame, which is the actual frame.
    ret, frame = cap.read()
    # checks if a frame was successfully read. 
    # If not (which means we have reached the end of the video), the loop breaks and the program will exit the video processing.
    if not ret:
        break
    # converts the BGR color image 'frame' into a grayscale image 'gray'. This is a common preprocessing step for many computer vision tasks.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # background subtractor (backSub) is used to apply background subtraction to the grayscale frame (gray). 
    # This results in a binary mask fgMask where foreground objects are represented as white pixels and the background is black.
    # fgMask acts as a binary mask highlighting the areas of motion or change in the current frame compared to the background model
    fgMask = backSub.apply(gray)
    # finds contours (continuous lines that form the boundaries of objects in an image) in the binary mask fgMask. 
    # also returns a list of contours
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # initializes an empty list 'current_detections' to store the detected centers of objects in the current frame.
    current_detections = []

    #iterates over each contour found in fgMask
    for contour in contours:
        
        if cv2.contourArea(contour) > 100:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour) # calculates the bounding rectangle around it using cv2.boundingRect(). 
            center = (int(x+w/2), int(y+h/2)) # The center of this bounding rectangle is calculated
            current_detections.append(center) # center is added to the current_detections list. 
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # a red circle is drawn at the center of each detected object on the original frame 'frame'
    # For each detection, it calculates the Euclidean distance btwn that detection and the last known pos of each tracked obj (planaria_tracks). 
    # It finds the closest tracked object (closest_id) to the current detection.
    for detection in current_detections: # iterates over each detection (center of a detected object)
        closest_id = None
        min_distance = float('inf')
        for planaria_id, positions in planaria_tracks.items():
            last_position = positions[-1]
            distance = np.linalg.norm(np.array(last_position) - np.array(detection))
            if distance < min_distance:
                min_distance = distance
                closest_id = planaria_id
    # If a closest tracked object is found and it's close enough (distance less than 50 pixels), -
    # -the current detection is added to the tracked object's positions list, and its speed is calculated-
    # -based on the distance traveled between the last two positions and the frame rate (fps). 
    # If no close object is found, a new tracking entry is created for the current detection (planaria_id_counter), and its speed is initialized to 0.
        if closest_id is not None and min_distance < 50:  # Threshold for "close enough"
            planaria_tracks[closest_id].append(detection)
            # Calculate speed
            if len(planaria_tracks[closest_id]) > 1:
                # Speed = distance / time
                # Distance is the Euclidean distance between the last two positions
                # Time is 1 frame duration (1/fps)
                speed = min_distance * fps  # pixels per second
                planaria_speeds[closest_id] = speed
        else:
            planaria_tracks[planaria_id_counter] = [detection]
            planaria_speeds[planaria_id_counter] = 0  # Initial speed is 0
            planaria_id_counter += 1

    for planaria_id, positions in planaria_tracks.items():
        if len(positions) > 1:
            total_distance = 0
            for i in range(1, len(positions)):
                total_distance += np.linalg.norm(np.array(positions[i - 1]) - np.array(positions[i]))
            cv2.putText(frame, f"ID: {planaria_id} Distance: {total_distance:.2f}px", positions[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(f"ID: {planaria_id} - Total Distance: {total_distance:.2f}px")  # Display total distance

        for i in range(1, len(positions)):
            cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

