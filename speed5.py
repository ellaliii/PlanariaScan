import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture('ControlStable.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Initialize tracking information
planaria_tracks = {}
planaria_id_counter = 0
planaria_start_frames = {}  # Track the start frame for each planaria

frame_count = 0  # Initialize frame counter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # Increment frame counter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgMask = backSub.apply(gray)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x+w/2), int(y+h/2))
            current_detections.append(center)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Mark center

    for detection in current_detections:
        closest_id = None
        min_distance = float('inf')
        for planaria_id, positions in planaria_tracks.items():
            last_position = positions[-1]
            distance = np.linalg.norm(np.array(last_position) - np.array(detection))
            if distance < min_distance:
                min_distance = distance
                closest_id = planaria_id

        if closest_id is not None and min_distance < 50:  # Threshold for "close enough"
            planaria_tracks[closest_id].append(detection)
            planaria_tracks[closest_id][1] += min_distance  # Update total distance
        else:
            planaria_tracks[planaria_id_counter] = [detection, 0]  # Initialize total distance
            planaria_start_frames[planaria_id_counter] = frame_count  # Record start frame
            planaria_id_counter += 1

    # Display logic for travel time and print to console
    for planaria_id, data in planaria_tracks.items():
        data = planaria_tracks[planaria_id]
        if len(data[0]) > 1:
            start_frame = planaria_start_frames[planaria_id]
            travel_time = (frame_count - start_frame) / fps  # Travel time in seconds
            total_distance = data[1]
            total_speed = total_distance / travel_time if travel_time > 0 else 0
            text = f"ID: {planaria_id} Time: {travel_time:.2f}s Distance: {total_distance:.2f}px Speed: {total_speed:.2f}px/s"
            position = tuple(np.array(data[0][-1], dtype=int))  # Convert position to integer tuple
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(f"ID: {planaria_id} - Travel Time: {travel_time:.2f}s Total Distance: {total_distance:.2f}px Total Speed: {total_speed:.2f}px/s")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

