import itertools
import logging
import pandas as pd
import os

# Parameters for combinations
units_options = [
    [50],
    [100, 50],
    [100, 100, 100]
]
dropout_ratios = [0.1, 0.3]
batch_sizes = [16, 32]
max_slices_options = [2500, 5000]

# Generate all combinations
param_combinations = itertools.product(units_options, dropout_ratios, batch_sizes, max_slices_options)

# Output directory for generated Python files
output_dir = "./generated_scripts"
os.makedirs(output_dir, exist_ok=True)

# Loop over all combinations and create separate Python files
i = 0
for units, dropout_ratio, batch_size, max_slices in param_combinations:
    i += 1
    # Generate a unique filename based on parameters
    filename = f"rnn{i}.py"
    script_path = os.path.join(output_dir, filename)

    # Create the Python code as a string
    code = f"""
import logging
import pandas as pd
import os
from main import *

if __name__ == '__main__':
    logging.info("Starting")
    # Call the function to set the seed
    set_seed(42)

    mapping = {{
        '10': 8.5,
        '5': 6.5,
        '0': 2
    }}

    dataset_name = "{dataset_name}"
    dataset_dir  = "{dataset_dir}"
    dataset_path = os.path.join(dataset_dir, dataset_name)
    df = pd.read_csv(dataset_path)
    df['target'] = df['target'].astype(str)
    df['target'] = df['filename'].apply(lambda x: mapping.get(x.split('_')[2]))

    # Create and test model
    m = Model(
        units={units},
        dropout_ratio={dropout_ratio},
        kernel_regularizer_l2=0,
        kernel_regularizer_l1=0,
        batch_size={batch_size},
        num_classes=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        epochs=1000,
        patience=100,
        max_slices={max_slices},
        mapping=mapping,
        dataset_name=dataset_name
    )
    
    m.test_model(df)
    """

    # Write the generated code to the file
    with open(script_path, "w") as script_file:
        script_file.write(code)

    logging.info(f"Generated script: {script_path}")
