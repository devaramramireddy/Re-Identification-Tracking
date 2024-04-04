import cv2
import numpy as np
import torch

class PreprocessImage:
    def __init__(self, config):
        self.config = config

    def preprocess_image(self, image, target_size):
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
        self.padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # Image enhancements (optional)
        # padded_image = cv2.equalizeHist(padded_image)

        # Normalize
        self.padded_image = self.padded_image.astype('float32') / 255.0
        self.padded_image = torch.tensor(self.padded_image).permute(2, 0, 1).unsqueeze(0)

        return self.padded_image


class ImagePreprocessor:
    def __init__(self, config):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.config = config

    def color_normalization(self, image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        lab_image[:, :, 0] = cv2.normalize(lab_image[:, :, 0], None, 0, 255, cv2.NORM_MINMAX)
        normalized_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
        return normalized_image

    def background_subtraction(self, image):
        foreground_mask = self.bg_subtractor.apply(image)
        result = cv2.bitwise_and(image, image, mask=foreground_mask)
        return result

    def noise_reduction(self, image, k_size=3):
        blurred_image = cv2.GaussianBlur(image, (k_size, k_size), 0)
        return blurred_image

    def histogram_equalization(self, image):
        if len(image.shape) == 3:
            # Convert color image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization to each channel
            equalized_image = cv2.equalizeHist(gray_image)
            # Convert back to color
            equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
        else:
            # Apply histogram equalization to grayscale image
            equalized_image = cv2.equalizeHist(image)

        return equalized_image

    def extract_thermal_values(self, thermal_image):
        # thermal_image = cv2.imread(thermal_image_path, cv2.IMREAD_GRAYSCALE)

        if thermal_image is None:
            print("Error: Unable to read the thermal image.")
            return None

        thermal_values = thermal_image.flatten()
        return thermal_values, thermal_image

    def threshold_person(self, thermal_image, threshold_value):
        # thermal_image = cv2.imread(thermal_image_path, cv2.IMREAD_GRAYSCALE)

        if thermal_image is None:
            print("Error: Unable to read the thermal image.")
            return None

        # Apply thresholding to separate person from background
        _, binary_mask = cv2.threshold(thermal_image, threshold_value, 255, cv2.THRESH_BINARY)

        return binary_mask

    def selective_roi_processing(self, image, roi_coords):
        x, y, w, h = roi_coords
        roi = image[y:y + h, x:x + w]
        # Apply specific preprocessing to this ROI
        roi_processed = self.apply_clahe(roi)
        image[y:y + h, x:x + w] = roi_processed
        return image

    def process_image(self, image):
        preprocessing_config = self.config.get('preprocessing', {})
        input_type = self.config.get('input_type')
        processed_image = image.copy()
        if input_type == "THERMAL":
            threshold_value = 125
            thermal_values, original_thermal_image = self.extract_thermal_values(processed_image)
            if thermal_values is not None:
                #advanced_denoising
                processed_image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

                # #apply_clahe
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # processed_image = clahe.apply(processed_image)


                #processed_image = processed_image/255.0
                #processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
                # Apply thresholding to distinguish person from background
                binary_mask = self.threshold_person(processed_image, threshold_value)

                # Set background values to zero explicitly
                person_thermal_values = original_thermal_image * binary_mask
                person_thermal_values[binary_mask == 0] = 0

                processed_image = self.histogram_equalization(person_thermal_values.astype(np.uint8))
                # processed_image = cv2.cvtColor(histeq, cv2.COLOR_GRAY2BGR)

        else:
            if preprocessing_config.get('histogram_equalization', False):
                processed_image = self.histogram_equalization(processed_image)
            if preprocessing_config.get('color_normalization', False):
                processed_image = self.color_normalization(processed_image)
            if preprocessing_config.get('background_subtraction', False):
                processed_image = self.background_subtraction(processed_image)
            if preprocessing_config.get('noise_reduction', False):
                processed_image = self.noise_reduction(processed_image)

        return processed_image