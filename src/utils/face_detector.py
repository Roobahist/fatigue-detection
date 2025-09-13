from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = (
    "/home/mojtaba.kananisarche/Zh/src/utils/face_landmarker_v2_with_blendshapes.task"
)


class FaceDetector:
    """
    A class for detecting faces and facial landmarks using MediaPipe.
    """

    def __init__(
        self,
        output_face_blendshapes: bool = False,
        output_facial_transformation_matrixes: bool = False,
        num_faces: int = 1,
    ):
        """
        Initializes the FaceDetector with specified options.

        Args:
            output_face_blendshapes: Whether to output face blendshapes.
            output_facial_transformation_matrixes: Whether to output facial transformation matrices.
            num_faces: The maximum number of faces to detect.
        """
        self.output_face_blendshapes = output_face_blendshapes
        self.output_facial_transformation_matrixes = (
            output_facial_transformation_matrixes
        )
        self.num_faces = num_faces
        self.landmarker = self._initialize_landmarker()

    def _initialize_landmarker(self) -> mp.tasks.vision.FaceLandmarker:
        """
        Initializes the MediaPipe FaceLandmarker.

        Returns:
            The initialized FaceLandmarker object.
        """
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=self.output_face_blendshapes,
            output_facial_transformation_matrixes=self.output_facial_transformation_matrixes,
            num_faces=self.num_faces,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(
        self, image: np.ndarray
    ) -> Optional[mp.tasks.vision.FaceLandmarkerResult]:
        """
        Detects faces and facial landmarks in the given image.

        Args:
            image: The input image as a NumPy array (BGR format).
        
        Returns:
            The detection result containing face landmarks, blendshapes, and transformation matrices.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a valid NumPy array.")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Perform face landmark detection
        detection_result = self.landmarker.detect(mp_image)
        
        return detection_result