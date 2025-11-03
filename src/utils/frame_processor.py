import os

from deepface import DeepFace as DF


class FrameProcessor:
    def __init__(self):
        pass

    def represent(self, *args, **kwargs):
        return DF.represent(*args, **kwargs)


# todo : make it compatible with Facenet and other models
# FP = FrameProcessor()
# frame_embedding = [
#     round(x, 3)
#     for x in FP.represent(
#         img_path=frame,
#         model_name=embedding_model,
#         enforce_detection=False,
#         detector_backend="skip",
#         normalization=embedding_normalization,
#     )[0]["embedding"]
# ]
