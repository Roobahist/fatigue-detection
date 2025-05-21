if __name__ == "__main__":
    import os
    import tracemalloc

    import memory_profiler
    import objgraph as o

    from Zh.src.process.face_video_processor import *
    from Zh.src.utils.memory import print_all_memory as vm

    pid = os.getpid()
    print(pid)

    # breakpoint()

    input_folder = "/home/mojtaba.kananisarche/Zh/data/UTA/1-raw_30fps"
    # output_folder='/work/long_lab/Mojtaba/data/UTA/1.3.1-augmentated_embedded'
    output_folder='/home/mojtaba.kananisarche/Zh/temp/out2'
    # output_folder = "/home/mojtaba.kananisarche/Zh/temp/out"
    vp = VideoProcessor(
        input_folder=input_folder, output_folder=output_folder, use_tqdm=False
    )
    augmentations = {
        "gaussianblur": [{"kernel_size": 3}, {"kernel_size": 5}],
        "brightness": [{"alpha": 1.2, "beta": 10}, {"alpha": 0.8, "beta": -10}],
        "contrast": [{"alpha": 1.5, "beta": 0}, {"alpha": 0.7, "beta": 0}],
        "flip": [
            {"flip_code": 1},  # Vertical flip
        ],
        "grayscale": [{}],  # No parameters needed
        "rotate": [{"angle": 5}, {"angle": -5}],
    }
    vp.process_multiple_videos(
        # video_files=[],  # all
        video_files = ['1_06_5_30fps.mp4'],
        # video_files=os.listdir(input_folder)[:2],
        parallel=True,
        start_second=0,
        end_second=None,
        # end_second=2,
        augmentations=augmentations,
        save=True,
        max_workers=15,
        find_embedding=True,
        embedding_model='Facenet'
    )
