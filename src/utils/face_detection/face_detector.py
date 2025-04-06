import mediapipe as mp
import numpy as np
from typing import Optional
import cv2

MODEL_PATH = '/home/mojtaba.kananisarche/Zh/src/utils/face_detection/face_landmarker_v2_with_blendshapes.task'

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
        self.output_facial_transformation_matrixes = output_facial_transformation_matrixes
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