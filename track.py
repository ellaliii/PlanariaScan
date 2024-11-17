import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture('52.mp4')
#cap = cv2.VideoCapture('ControlStable.mp4')
#cap = cv2.VideoCapture('Etoh5.mp4')
#cap = cv2.VideoCapture('EtOH001.mp4')
#cap = cv2.VideoCapture('C60-0036.mp4')
#cap = cv2.VideoCapture('C60-0006.mp4')
#cap = cv2.VideoCapture('Caffeine0001.mp4')
#cap = cv2.VideoCapture('Caffeine00001.mp4')

backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# This will hold the positions of planaria in the format {id: [(x1, y1), (x2, y2), ...], ...}
planaria_tracks = {}
planaria_id_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgMask = backSub.apply(gray)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Temporary storage for current frame's detections
    current_detections = []

    for contour in contours:
        #if cv2.contourArea(contour) > 100:  # Filter out small contours
        if cv2.contourArea(contour) > 20:  # Filter out small contours    
            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x+w/2), int(y+h/2))
            current_detections.append(center)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Mark center

    # Track or update planaria positions
    # For simplicity, this example assumes planaria do not cross paths closely
    # A more robust method would be required for dealing with close crossings or occlusions
    for detection in current_detections:
        # Find the closest track (if any) to this detection
        closest_id = None
        min_distance = float('inf')
        for planaria_id, positions in planaria_tracks.items():
            last_position = positions[-1]
            distance = np.linalg.norm(np.array(last_position) - np.array(detection))
            if distance < min_distance:
                min_distance = distance
                closest_id = planaria_id
        
        # If a track is close enough, update it; otherwise, start a new track
        #PY!if closest_id is not None and min_distance < 50:  # Threshold for "close enough"
        if closest_id is not None and min_distance < 15:  # Threshold for "close enough"    
            planaria_tracks[closest_id].append(detection)
        else:
            planaria_tracks[planaria_id_counter] = [detection]
            planaria_id_counter += 1

    # Optional: Visualize tracks
    for planaria_id, positions in planaria_tracks.items():
        for i in range(1, len(positions)):
            #cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 8) #blue 
            #cv2.line(frame, positions[i - 1], positions[i], (0, 255, 0), 8) #green 
            cv2.line(frame, positions[i - 1], positions[i], (0, 255, 255), 2) #yellow 
            #cv2.line(frame, positions[i - 1], positions[i], (0, 255, 255), 8)
            #skyblue cv2.line(frame, positions[i - 1], positions[i], (230, 216, 173), 2)
            #cv2.line(frame, positions[i - 1], positions[i], (80, 0, 80), 8) #purple 

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#yellow (255, 255, 0)
#red (255, 0, 0)
#green  (0, 255, 0)  C60
#blue (0, 0, 255)
#sky blue (173, 216, 230)