import numpy as np
import cv2


class RiceLeafAnalyzer:
	def __init__(self):
		# Improved HSV ranges for better detection
		# Nitrogen deficiency: Yellowing (chlorosis)
		self.yellow_hsv_range = ((15, 50, 50), (35, 255, 255))
		
		# Phosphorus deficiency: Dark green/purple discoloration
		self.purple_hsv_range = ((120, 30, 30), (160, 255, 255))
		
		# Potassium deficiency: Brown/rust spots and edges
		self.brown_hsv_range = ((0, 30, 30), (20, 255, 200))
		
		# Additional ranges for better detection
		self.dark_green_range = ((40, 50, 20), (80, 255, 150))  # Phosphorus dark green
		self.rust_range = ((0, 50, 50), (15, 255, 255))        # Potassium rust
		
		# Improved thresholds based on actual deficiency patterns
		self.min_ratio_thresholds = {
			"Nitrogen": 0.05,    # Lower threshold for yellowing
			"Phosphorus": 0.03,  # Lower threshold for purple/dark green
			"Potassium": 0.03,   # Lower threshold for brown/rust
		}

	def analyze_color_features(self, image_rgb: np.ndarray) -> dict:
		"""Return comprehensive color ratios indicative of each deficiency."""
		if image_rgb is None or image_rgb.size == 0:
			raise ValueError("Empty image passed to analyze_color_features")

		# Convert to HSV for color masking
		hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
		total_pixels = float(image_rgb.shape[0] * image_rgb.shape[1])

		# Basic color masks
		yellow_mask = cv2.inRange(hsv, self.yellow_hsv_range[0], self.yellow_hsv_range[1])
		purple_mask = cv2.inRange(hsv, self.purple_hsv_range[0], self.purple_hsv_range[1])
		brown_mask = cv2.inRange(hsv, self.brown_hsv_range[0], self.brown_hsv_range[1])
		
		# Additional masks for better detection
		dark_green_mask = cv2.inRange(hsv, self.dark_green_range[0], self.dark_green_range[1])
		rust_mask = cv2.inRange(hsv, self.rust_range[0], self.rust_range[1])

		# Calculate ratios
		features = {
			"yellow_ratio": float(np.count_nonzero(yellow_mask)) / total_pixels,
			"purple_ratio": float(np.count_nonzero(purple_mask)) / total_pixels,
			"brown_ratio": float(np.count_nonzero(brown_mask)) / total_pixels,
			"dark_green_ratio": float(np.count_nonzero(dark_green_mask)) / total_pixels,
			"rust_ratio": float(np.count_nonzero(rust_mask)) / total_pixels,
		}
		
		# Calculate combined deficiency indicators
		features["nitrogen_score"] = features["yellow_ratio"]
		features["phosphorus_score"] = max(features["purple_ratio"], features["dark_green_ratio"])
		features["potassium_score"] = max(features["brown_ratio"], features["rust_ratio"])
		
		return features

	def detect_deficiency(self, image_rgb: np.ndarray) -> str:
		"""Predict deficiency class with improved logic."""
		features = self.analyze_color_features(image_rgb)
		
		# Use combined scores for better detection
		scores = {
			"Nitrogen": features["nitrogen_score"],
			"Phosphorus": features["phosphorus_score"],
			"Potassium": features["potassium_score"],
		}

		# Find the highest scoring deficiency
		best_class = max(scores, key=scores.get)
		best_score = scores[best_class]
		
		# If the best score is very low, check for healthy vs deficiency
		if best_score < 0.01:  # Very low color indicators
			# Check if image is mostly green (healthy)
			hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
			green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
			green_ratio = float(np.count_nonzero(green_mask)) / (image_rgb.shape[0] * image_rgb.shape[1])
			
			if green_ratio > 0.7:  # Mostly green
				return "Healthy"
			else:
				return best_class  # Return best guess even if low confidence
		
		# Return the deficiency with highest score
		return best_class