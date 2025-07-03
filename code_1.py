#importing libraries
import cv2
import os
import numpy as np
import imutils
from datetime import datetime

#establishing sources and other variables
video_source = "video.mp4"  
new_dir = "Suspects"
face_padding = 30
frame_width = 820
save_images = True

#creating directory
os.makedirs(new_dir, exist_ok=True)

#face detection using haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

#processing frames by drawing rectangles and saving the cropped images
def detect_and_draw_faces(frame, count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        x1 = max(x - face_padding, 0)
        y1 = max(y - face_padding, 0)
        x2 = min(x + w + face_padding, frame.shape[1])
        y2 = min(y + h + face_padding, frame.shape[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) #drawing a red rectangle

        #saving the cropped face
        if save_images:
            face_crop = frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(new_dir, f"Suspect_{count}_{timestamp}.png") #saving images with a count and timestamp
            cv2.imwrite(filename, face_crop)
            print(f"Saved: {filename}")
            count += 1

    return frame, count


def main():
    print("Initializing Cam Cropify. Starting video stream from source...") #starting the program
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("ERROR. Can't open video source.") #error handling 
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame.") #completion message
            break

        frame = imutils.resize(frame, width=frame_width) #resizing frames
        frame, count = detect_and_draw_faces(frame, count)

        cv2.imshow("Cam Cropify", frame) #streaming source video

        if cv2.waitKey(1) & 0xFF == ord("q"): #terminating the streaming when "q" is pressed
            print("Terminating code...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()