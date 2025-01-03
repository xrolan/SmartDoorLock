import face_recognition
import os
import sys
import cv2
import numpy as np
import math
import pygame
import re

# Initialize pygame for sound alerts
pygame.mixer.init()
pygame.mixer.music.load("./ringbell/audiomass-output1.mp3")
pygame.mixer.music.set_volume(1.0)

def play_ring_sound():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_rec_color = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.encode_faces()

    def encode_faces(self):
        directory = 'faces'
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                print(f"Processing directory: {subdir}")
                for image_name in os.listdir(subdir_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                        image_path = os.path.join(subdir_path, image_name)
                        try:
                            face_image = face_recognition.load_image_file(image_path)
                            # Convert to RGB if needed
                            if face_image.shape[2] == 4:  # If image has alpha channel
                                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
                            elif len(face_image.shape) == 2:  # If image is grayscale
                                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
                            
                            # Get face locations first
                            face_locations = face_recognition.face_locations(face_image)
                            if face_locations:
                                # Only get encodings if faces were found
                                face_encoding = face_recognition.face_encodings(face_image, face_locations)[0]
                                self.known_face_encodings.append(face_encoding)
                                self.known_face_names.append(os.path.splitext(image_name)[0])
                            else:
                                print(f"No faces found in {image_name}")
                        except Exception as e:
                            print(f"Error processing {image_name}: {str(e)}")
                    else:
                        print(f"Skipped non-image file: {image_name}")
        print("Loaded faces:", self.known_face_names)

    def add_new_face(self, frame, name):
        face_dir = os.path.join("faces", name)
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)

        face_image_path = os.path.join(face_dir, f"{name}.jpg")
        cv2.imwrite(face_image_path, frame)
        self.encode_faces()

    def run_recognition(self):
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        total_frame = 0

        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        iter_time = 0
        user_init = 0

        while True:

            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame.")
                break

            total_frame+=1 

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Ensure the frame is in RGB format
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Get face locations
                self.face_locations = face_recognition.face_locations(rgb_small_frame)

                # Check for detected faces
                if len(self.face_locations) == 0:
                    threshold_value = 30
                    blank_frame_condition = np.mean(frame) < threshold_value
                    if blank_frame_condition:
                        TN += 1
                    else:
                        FN += 1
                    self.face_names = []
                    self.face_rec_color = []
                else:
                    try:
                        # Get face encodings for found faces
                        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                        self.face_names = []
                        self.face_rec_color = []

                        for face_encoding in self.face_encodings:
                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                            name = "Unknown"
                            confidence = 'Unknown'
                            user = '???'

                            if len(self.known_face_encodings) > 0:
                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)

                                if matches[best_match_index]:
                                    name = self.known_face_names[best_match_index]
                                    confidence = face_confidence(face_distances[best_match_index])

                                if confidence != 'Unknown':
                                    conf_percent = float(confidence.rstrip('%'))
                                    if conf_percent >= 95:
                                        TP += 1
                                        user = "Home Owner (Unlocked)"
                                        user_init = 1
                                        rec_color = (255, 0, 0)
                                    else:
                                        FP += 1
                                        user = "Stranger (Locked)"
                                        user_init = 0
                                        rec_color = (0, 0, 255)
                                else:
                                    user = "Stranger (Locked)"
                                    user_init = 0
                                    rec_color = (0, 0, 255)

                            name = re.sub(r'\d', '', name)
                            self.face_names.append(f'{user} : {name} ({confidence})')
                            self.face_rec_color.append(rec_color)

                    except Exception as e:
                        print(f"Error during face recognition: {str(e)}")
                        continue

            self.process_current_frame = not self.process_current_frame

            if user_init == 0 and iter_time > 4:
                play_ring_sound()

            if user_init == 1:
                iter_time = 0
            else:
                iter_time += 1

            # Calculate and display accuracy
            total_predictions = TP + TN + FP + FN
            accuracy = (TP + TN) / total_predictions if total_predictions > 0 else 0
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display annotations
            for (top, right, bottom, left), name, rec_color in zip(self.face_locations, self.face_names, self.face_rec_color):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), rec_color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), rec_color, -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            cv2.imshow('Face Recognition', frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

        # Print metrics eval
        print(f"Total Frame: {total_frame}")
        print(f"TP: {TP}")
        print(f"TN: {TN}")
        print(f"FN: {FN}")
        print(f"FP: {FP}")

        total = TP + TN + FP + FN
        if total > 0:
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            accuracy = (TP + TN) / total
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print("\nFinal Metrics:")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            print(f"F1 Score: {f1_score:.2%}")

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()