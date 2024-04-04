import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchreid.reid.utils import FeatureExtractor
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from preprocessing import PreprocessImage, ImagePreprocessor

from config import Config
import os

class PersonTracker:
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_type = config["input_type"]
        self.video_path = config["video_path"]
        self.yolo = YOLO(config["yolo_model_path"])
        self.extractor = FeatureExtractor(model_name=config["reid_model_name"], model_path=config["reid_model_path"], device=self.device.type)
        self.person_id_counter = config["tracking_id"]
        self.tracked_persons = {}  # Stores person ID and their features and Kalman filter
        self.similarity_threshold_yolo = config["similarity_threshold_yolo"]
        self.similarity_threshold_reid = config["similarity_threshold_reid"]
        self.output_path = config["output_path"]
        self.output_video_path = os.path.join(self.output_path, "output_video.mp4")
        self.output_video = None  # VideoWriter object for saving the processed frames as a video
        self.display_window_name = "Processed Frame"  # Name of the display window
        self.tracking_id = config["tracking_id"]
        self.target_size = (128, 256)
        self.preprocess = PreprocessImage(config)
        self.image_preprocess = ImagePreprocessor(config)


    def initialize_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P *= 1000
        kf.R = np.eye(2) * 10
        kf.Q = Q_discrete_white_noise(dim=4, dt=1, var=0.01)
        return kf

    def preprocess_image(self, image, target_size=(128, 256)):
        # Resize with padding
        h, w = image.shape[:2]
        scale = min(target_size[1] / h, target_size[0] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_image = cv2.resize(image, (new_w, new_h))

        delta_w = target_size[0] - new_w
        delta_h = target_size[1] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # Image enhancements (optional)
        #padded_image = cv2.equalizeHist(padded_image)

        # Normalize
        padded_image = padded_image.astype('float32') / 255.0
        padded_image = torch.tensor(padded_image).permute(2, 0, 1).unsqueeze(0)
        return padded_image

    def get_person_features(self, crop):
        preprocessed_crop = self.preprocess.preprocess_image(crop, target_size=self.target_size)
        # preprocessed_crop = self.image_preprocess.process_image(crop)
        # cv2_image_rgb = cv2.cvtColor(preprocessed_crop, cv2.COLOR_BGR2RGB)
        # input_tensor = torch.tensor(cv2_image_rgb, dtype=torch.float32) / 255.0
        # input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        # input_tensor = input_tensor.to(self.device)
        features = self.extractor(preprocessed_crop)
        return features

    def find_matching_person(self, new_features):
        for person_id, data in self.tracked_persons.items():
            similarity = torch.nn.functional.cosine_similarity(new_features, data['features'], dim=1)
            if similarity > self.similarity_threshold_reid:
                return person_id
        return None

    def initialize_video_writer(self, frame):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            30,  # Frames per second (adjust as needed)
            (frame.shape[1], frame.shape[0]),  # Frame size
        )

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.output_video is None:
                self.initialize_video_writer(frame)

            processed_frame = self.update_persons(frame)

            # Display the processed frame
            cv2.imshow(self.display_window_name, processed_frame)

            # Write the processed frame to the output video
            self.output_video.write(processed_frame)

            # Add a delay (in milliseconds) to control frame display speed
            delay = 10  # Adjust as needed
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        # Release the video writer when done
        if self.output_video is not None:
            self.output_video.release()

        # Destroy the display window
        cv2.destroyWindow(self.display_window_name)

    def update_persons(self, frame):
        results = self.yolo(frame)
        current_frame_ids = set()  # IDs assigned in the current frame

        for x1, y1, x2, y2, conf, cls_id in results[0].boxes.data:
            if cls_id == 0 and conf > self.similarity_threshold_yolo:  # Class '0' for person
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                features = self.get_person_features(crop)

                best_match_id = None
                highest_similarity = 0

                # Check for the best match among existing tracked persons
                for person_id, data in self.tracked_persons.items():
                    if person_id not in current_frame_ids:
                        similarity = torch.nn.functional.cosine_similarity(features, data['features'], dim=1)
                        if similarity > self.similarity_threshold_reid and similarity > highest_similarity:
                            best_match_id = person_id
                            highest_similarity = similarity

                # Assign a new ID if no suitable match is found or if the ID is already used in this frame
                if best_match_id is None:
                    best_match_id = self.person_id_counter
                    self.person_id_counter += 1
                    self.tracked_persons[best_match_id] = {'features': features, 'kalman_filter': self.initialize_kalman_filter()}

                current_frame_ids.add(best_match_id)

                # Update Kalman filter for this person
                kf = self.tracked_persons[best_match_id]['kalman_filter']
                kf.predict()
                measurement = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2])
                kf.update(measurement)

                # Draw bounding box and ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {best_match_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

                # Highlight the tracked person (if applicable)
                if best_match_id == self.tracking_id:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)  # Red box for tracked person

        return frame


    #def update_persons(self, frame):
    #    results = self.yolo(frame)
    #    current_frame_ids = set()
    #    potential_matches = {}
    #    for x1, y1, x2, y2, conf, cls_id in results[0].boxes.data:
    #        if cls_id == 0 and conf > self.similarity_threshold_yolo:  # Class '0' for person
    #            crop = frame[int(y1):int(y2), int(x1):int(x2)]
    #            features = self.get_person_features(crop)
    #            person_id = self.find_matching_person(features)
#
    #            if person_id is None:
    #                person_id = self.person_id_counter
    #                self.person_id_counter += 1
    #                self.tracked_persons[person_id] = {'features': features,
    #                                                   'kalman_filter': self.initialize_kalman_filter()}
#
    #            current_frame_ids.add(person_id)  # Mark this ID as used in the current frame
#
    #            # Update Kalman filter for this person
    #            kf = self.tracked_persons[person_id]['kalman_filter']
    #            kf.predict()
    #            measurement = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2])
    #            kf.update(measurement)
#
    #            # Draw bounding box and ID
    #            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #            cv2.putText(frame, f'ID: {person_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                        (0, 255, 0), 2)
#
    #            # If this is the person we are tracking, highlight it
    #            if person_id == self.tracking_id:
    #                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),
    #                              3)  # Red box for tracked person
#
    #    return frame
