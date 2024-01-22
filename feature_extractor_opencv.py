import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Initialize the HOG descriptor
        self.hog = cv2.HOGDescriptor()

    def extract(self, img_path):
        """
        Extract HOG features from an input image
        Args:
            img_path: Path to the image file

        Returns:
            feature (np.ndarray): HOG feature vector
        """
        # Read the image
        print("extractor img_path", img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image is loaded
        if image is None:
            raise ValueError("Image not found or path is incorrect")

        # Resize image to match HOG descriptor size
        image = cv2.resize(image, self.hog.winSize)

        # Compute HOG descriptors/features
        hog_features = self.hog.compute(image)

        # Flatten the feature vector and normalize
        feature_vector = hog_features.flatten()
        normalized_feature_vector = feature_vector / np.linalg.norm(feature_vector)

        return normalized_feature_vector