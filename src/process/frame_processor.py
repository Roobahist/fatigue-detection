import os

from deepface import DeepFace as DF


class FrameProcessor:
    def __init__(self):
        pass

    def represent(self, *args, **kwargs):
        return DF.represent(*args, **kwargs)
