import cv2
import numpy as np

cap = cv2.VideoCapture('52.mp4')
if not cap.isOpened():
    print("Error opening video stream or file")

backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

planaria_tracks = {}
planaria_id_counter = 0
inactive_threshold = 10  # Frames after which a track is considered inactive if not updated
merge_distance_threshold = 50  # Distance threshold for considering merging tracks
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgMask = backSub.apply(gray)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x + w / 2), int(y + h / 2))
            current_detections.append((center, frame_count))  # Include frame count for each detection

    # Update and merge tracks
    updated_tracks = {}
    for detection in current_detections:
        detected_center, _ = detection
        closest_id = None
        min_distance = float('inf')
        
        for planaria_id, (positions, last_seen) in planaria_tracks.items():
            last_position, _ = positions[-1]
            distance = np.linalg.norm(np.array(last_position) - np.array(detected_center))
            if distance < min_distance:
                min_distance = distance
                closest_id = planaria_id

        if closest_id is not None and min_distance < merge_distance_threshold:
            updated_tracks[closest_id] = (planaria_tracks[closest_id][0] + [detection], frame_count)
        else:
            updated_tracks[planaria_id_counter] = ([detection], frame_count)
            planaria_id_counter += 1

    # Check for inactive tracks and attempt to merge
    for id_a, (track_a, last_seen_a) in updated_tracks.items():
        if frame_count - last_seen_a > inactive_threshold:
            for id_b, (track_b, last_seen_b) in updated_tracks.items():
                if id_a != id_b:
                    # Check if tracks are close enough to merge at the point of last seen
                    if np.linalg.norm(np.array(track_a[-1][0]) - np.array(track_b[0][0])) < merge_distance_threshold:
                        # Merge track_b into track_a and update last_seen to the most recent
                        updated_tracks[id_a] = (track_a + track_b, max(last_seen_a, last_seen_b))
                        del updated_tracks[id_b]
                        break

    planaria_tracks = updated_tracks

    # Display current frame's detections and track count
    cv2.putText(frame, f"Total Planaria: {len(planaria_tracks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for planaria_id, (positions, _) in planaria_tracks.items():
        for position, _ in positions:
            cv2.circle(frame, position, 5, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
