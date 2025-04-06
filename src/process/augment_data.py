if __name__ == "__main__":
    from Zh.src.process.face_video_processor import *

    input_folder = "/home/mojtaba.kananisarche/Zh/data/UTA/1-raw_30fps"
    # output_folder='/home/mojtaba.kananisarche/Zh/temp/out'
    output_folder = "/work/long_lab/Mojtaba/augmented3"

    vp = VideoProcessor(
        input_folder=input_folder, output_folder=output_folder, use_tqdm=True
    )

    augmentations = {
        # "gaussianblur": [
        #     {"kernel_size": 3},
        #     {"kernel_size": 5}
        # ],
        "brightness": [{"alpha": 1.2, "beta": 10}, {"alpha": 0.8, "beta": -10}],
        "contrast": [{"alpha": 1.5, "beta": 0}, {"alpha": 0.7, "beta": 0}],
        "flip": [
            {"flip_code": 1},  # Vertical flip
        ],
        "grayscale": [{}],  # No parameters needed
        "rotate": [{"angle": 5}, {"angle": -5}],
    }

    vp.process_multiple_videos(
        video_files=[],
        parallel=True,
        start_second=0,
        end_second=None,
        augmentations=augmentations,
        save=True,
        max_workers=30,
    )
