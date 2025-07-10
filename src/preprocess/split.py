import os
import shutil

from Zh.src.utils.file_folder import create_folders

def copy_files_recursive(input_dir, output_dir):
    """
    Copies all files from input_dir to output_dir, preserving the directory structure.
    
    Args:
        input_dir (str): Path to the source directory.
        output_dir (str): Path to the destination directory.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    create_folders(
        [
            output_dir,

            os.path.join(output_dir, 'train'),
            os.path.join(output_dir, 'train', '0'),
            os.path.join(output_dir, 'train', '5'),
            os.path.join(output_dir, 'train', '10'),

            os.path.join(output_dir, 'val'),
            os.path.join(output_dir, 'val', '0'),
            os.path.join(output_dir, 'val', '5'),
            os.path.join(output_dir, 'val', '10'),

            os.path.join(output_dir, 'test'),
            os.path.join(output_dir, 'test', '0'),
            os.path.join(output_dir, 'test', '5'),
            os.path.join(output_dir, 'test', '10'),
        ]
    )
    
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist.")
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:

            filename_split = file.split('_')
            fold = filename_split[0]
            label = filename_split[2]

            if fold == '1':
                split = 'test'

            elif fold == '2':
                split = 'val'

            else: 
                split = 'train'

            source_path = os.path.join(root, file)
            target_path = os.path.join(output_dir, split, label, file)

            shutil.copy2(source_path, target_path)


if __name__ == "__main__":
    input_dir = "/work/long_lab/Mojtaba/data/UTA/1.4-augmentated_masked/cropped"
    output_dir = "/work/long_lab/Mojtaba/data/UTA/1.4-augmentated_masked_cropped_split1"
    
    copy_files_recursive(input_dir=input_dir, 
                         output_dir=output_dir)