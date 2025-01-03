import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import os
from pathlib import Path

class MultiReferenceFaceRecognition:
    def __init__(self):
        self.sift = cv2.SIFT_create(
            nfeatures=5000,
            nOctaveLayers=10, 
            contrastThreshold=0.01,
            edgeThreshold=10,
            sigma=1.6
        )

        self.match_counts = {}  # untuk ngetrack match per label
        self.total_frames = 0 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(keep_all=True, device=self.device)
        
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        
        self.ratio_threshold = 0.7
        self.min_matches = 2
        self.confidence_threshold = 30.0 
        self.is_reference_mode = True
        self.reference_database = []
        
    def preprocess_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return image
        
    def detect_face(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs = self.detector.detect(rgb_image)
        print(f"Detected boxes: {boxes}, probabilities: {probs}")
        
        if boxes is None or len(boxes) == 0:
            print("No face boxes detected.")
            return None
        
        confidence_threshold = 0.6
        confident_faces = [(box, prob) for box, prob in zip(boxes, probs) if prob > confidence_threshold]
        if not confident_faces:
            print("No confident faces detected.")
            return None
        
        # Select the largest face
        largest_face = max(confident_faces, key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))
        box = largest_face[0]
        
        # Ensure box dimensions are valid
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            print("Invalid face box dimensions.")
            return None
        
        return (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))

        
    def extract_features(self, image):
        processed_image = self.preprocess_image(image)
        face_region = self.detect_face(processed_image)
        
        if face_region is None:
            print("No face detected.")
            return None, None, None
        
        x, y, w, h = face_region
        roi = processed_image[y:y+h, x:x+w]
        
        # Ensure ROI is not empty
        if roi is None or roi.size == 0:
            print("ROI is empty after cropping.")
            return None, None, None
        
        try:
            keypoints, descriptors = self.sift.detectAndCompute(roi, None) #SIFT
            print(f"Keypoints: {len(keypoints) if keypoints else 0}, Descriptors: {descriptors.shape if descriptors is not None else None}")
            return keypoints, descriptors, roi
        except cv2.error as e:
            print(f"Error during SIFT computation: {e}")
            return None, None, None

        
    def load_reference_images(self, reference_dir):
        self.reference_database = []
        reference_dir = Path(reference_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for subdir in reference_dir.iterdir():
            if subdir.is_dir(): 
                label = subdir.name  # subdir
                for img_path in subdir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        print(f"Loading reference image: {img_path}")
                        ref_img = cv2.imread(str(img_path))
                        if ref_img is not None:
                            kp, desc, roi = self.extract_features(ref_img)
                            if kp and desc is not None:
                                self.reference_database.append({
                                    'keypoints': kp,
                                    'descriptors': desc,
                                    'roi': roi,
                                    'path': img_path,
                                    'label': label  # Store the label
                                })
                                print(f"Successfully added reference image with {len(kp)} keypoints")
                            else:
                                print(f"Failed to extract features from {img_path}")
                        else:
                            print(f"Failed to load {img_path}")
        print(f"Loaded {len(self.reference_database)} reference images")

        
    def match_faces(self, desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            return []
        # desc1 = cv2.normalize(desc1, None, alpha=0, beta=1, norm_type=cv2.NORM_L2)
        # desc2 = cv2.normalize(desc2, None, alpha=0, beta=1, norm_type=cv2.NORM_L2)
        try:
            matches = self.flann.knnMatch(desc1, desc2, k=2)
        except cv2.error as e:
            print(f"FLANN matching error: {e}")  # Debug
            return []
        good_matches = [m for m, n in matches if m.distance < self.ratio_threshold * n.distance]
        print(f"Good matches: {len(good_matches)}")  # Debug
        return good_matches
        
    def verify_matches(self, kp1, kp2, matches):
        if len(matches) < self.min_matches:
            return False, None
        try:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H is not None, mask
        except Exception as e:
            print(f"RANSAC error: {e}")  # Debug
            return False, None
        
    def calculate_confidence(self, matches, ref_kp, frame_kp, mask):
        if not matches or mask is None:
            return 0.0
        match_ratio = len(matches) / min(len(ref_kp), len(frame_kp))
        inlier_ratio = np.mean(mask) if mask is not None else 0
        distances = [m.distance for m in matches]
        avg_distance = np.mean(distances) if distances else 1.0
        distance_conf = 1.0 - min(avg_distance / 100.0, 1.0)
        confidence = (0.4 * match_ratio + 0.4 * inlier_ratio + 0.2 * distance_conf) * 100
        print(f"Match Ratio: {match_ratio}, Inlier Ratio: {inlier_ratio}, Avg Distance: {avg_distance}, Confidence: {confidence}")  # Debug
        return confidence
        
    def process_frame(self, frame):
        self.total_frames += 1  # Increment total frames
        frame_kp, frame_desc, frame_roi = self.extract_features(frame)
        if frame_kp is None:
            return frame, False, 0.0, []
        
        best_confidence = 0.0
        best_matches = None
        best_mask = None
        best_ref_data = None
        
        for ref_data in self.reference_database:
            matches = self.match_faces(ref_data['descriptors'], frame_desc)
            is_match, mask = self.verify_matches(ref_data['keypoints'], frame_kp, matches)
            if is_match:
                confidence = self.calculate_confidence(matches, ref_data['keypoints'], frame_kp, mask)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_matches = matches
                    best_mask = mask
                    best_ref_data = ref_data
        
        if best_confidence >= self.confidence_threshold and best_ref_data is not None:
            label = best_ref_data['label']
            self.match_counts[label] = self.match_counts.get(label, 0) + 1  # Increment match count for the label
            
            try:
                result_img = cv2.drawMatches(
                    best_ref_data['roi'], best_ref_data['keypoints'],
                    frame_roi, frame_kp,
                    best_matches, None,
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.putText(result_img, 
                            f"Label: {label}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
                cv2.putText(result_img, 
                            f"Match Confidence: {best_confidence:.1f}%", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
                return result_img, True, best_confidence, frame_kp
            
            except Exception as e:
                print(f"Error drawing matches: {e}")  # Debug
                return frame, False, 0.0, []
        
        return frame, False, 0.0, frame_kp

    def display_match_percentages(self):
        if self.total_frames == 0:
            return "No frames processed yet."
        
        percentages = {label: (count / self.total_frames) * 100 
                    for label, count in self.match_counts.items()}
        
        print("\nMatch Percentages:")
        for label, percentage in percentages.items():
            print(f"{label}: {percentage:.2f}%")
        print(f"Total Frames: {self.total_frames}")


# Usage example
def main():
    recognizer = MultiReferenceFaceRecognition()
    reference_dir = './faces'
    recognizer.load_reference_images(reference_dir)
    if len(recognizer.reference_database) == 0:
        print("No reference images loaded. Please add images to the reference_images directory.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open video capture")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_img, is_match, _, _ = recognizer.process_frame(frame)
            status = "MATCH" if is_match else "NO MATCH"
            color = (0, 255, 0) if is_match else (0, 0, 255)
            cv2.putText(result_img, status, (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Face Recognition', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recognizer.is_reference_mode = not recognizer.is_reference_mode
                print(f"Switched to {'Reference' if recognizer.is_reference_mode else 'Non-Reference'} Mode")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.display_match_percentages()  # Display percentages when exiting


if __name__ == "__main__":
    main()
