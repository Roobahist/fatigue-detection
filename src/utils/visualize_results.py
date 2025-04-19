from typing import Optional
from Zh.src.utils.file_folder import create_folders
import os
import json
import pandas as pd
import dataframe_image as dfi
import asyncio

async def table_results(
results_directory_path : str,
output_directory_path  : Optional[str] = None,
output_file_path_prefix: Optional[str] = 'results',
cmap                   : Optional[str] = "Greys",
reverse_cmap           : Optional[bool] = False,
maximum_tries          : Optional[int] = 5,
)->None: 
    
    KEYS_TO_COLLECT       = ['index', 'test', 'val', 'train_report', 'val_report', 'test_report']
    COLORS                = ['Accent','Blues','BrBG','BuGn','BuPu','CMRmap','Dark2','GnBu','Greens',
                             'Greys','OrRd','Oranges','PRGn','Paired','Pastel1','Pastel2','PiYG',
                             'PuBuGn','PuBu','PuOr','PuRd','Purples','RdBu','RdGy','RdPu','RdYlBu',
                             'RdYlGn','Reds','Set1','Set2','Set3','Spectral','Wistia','YlGnBu','YlGn',
                             'YlOrBr','YlOrRd','afmhot','autumn','binary','bone','brg','bwr','cividis',
                             'cool','coolwarm','copper','cubehelix','flag','gist_earth','gist_gray',
                             'gist_heat','gist_ncar','gist','gist_rainbow','gist_stern','gist_yarg',
                             'gnuplot2','gnuplot','gray','hot','hsv','inferno','jet','magma','nipy_spectral'
                             ,'ocean','pink','plasma','prism','rainbow','seismic','spring','summer','tab10',
                             'tab20','tab20b','tab20c','terrain','turbo','twilight','twilight_shifted','viridis','winter']

    output_directory_path = output_directory_path or results_directory_path
    create_folders(output_directory_path)
    output_file_name = None

    i = 0
    while output_file_name is None:
        i += 1
        if not os.path.exists(os.path.join(output_directory_path, f"{output_file_path_prefix}_{i}.png")):
            output_file_name = f"{output_file_path_prefix}_{i}.png"
    
    output_file_path = os.path.join(output_directory_path, output_file_name)

    data = []
    for file_name in os.listdir(results_directory_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(results_directory_path, file_name)
            with open(file=file_path, mode='r') as json_file:
                j = json.load(json_file)
                try:
                    data.append(
                        {
                            k:v for k, v in j.items() if k in KEYS_TO_COLLECT
                        }
                    )
                except KeyError as e:
                    print("Unable")
                    continue


    data = pd.DataFrame(data = data).dropna(how='all')
    try:
        data.index = data['index'].astype(int).values
        data       = data.sort_index()
        data       = data.drop('index', axis=1)
    except:
        print("No Index")
        pass

    folded_by_column = list(data['test'][0].keys())[0]

    multi_index = pd.MultiIndex.from_tuples([
    (f'Split ({folded_by_column})', 'Test'),
    (f'Split ({folded_by_column})', 'Validation'),
    ('Report', 'Train'),
    ('Report', 'Validation'),
    ('Report', 'Test'),
    ])
    data.columns = multi_index

    data[(f'Split ({folded_by_column})', 'Test')] = data[(f'Split ({folded_by_column})', 'Test')].apply(lambda x: x[folded_by_column])
    data[(f'Split ({folded_by_column})', 'Validation')]  = data[(f'Split ({folded_by_column})', 'Validation')].apply(lambda x: x[folded_by_column])

    data[('MAE', 'Train')] = data[('Report', 'Train')].apply(lambda x: round(x['overall']['mae'], 3))
    data[('MAE', 'Validation')]   = data[('Report', 'Validation')].apply(lambda x: round(x['overall']['mae'], 3))
    data[('MAE', 'Test')]  = data[('Report', 'Test')].apply(lambda x: round(x['overall']['mae'], 3))

    data[('Accuracy', 'Train')] = data[('Report', 'Train')].apply(lambda x: round(x['overall']['accuracy'], 3))
    data[('Accuracy', 'Validation')]   = data[('Report', 'Validation')].apply(lambda x: round(x['overall']['accuracy'], 3))
    data[('Accuracy', 'Test')]  = data[('Report', 'Test')].apply(lambda x: round(x['overall']['accuracy'], 3))

    data = data.drop('Report', axis=1, level=0)
    data.columns.names = [None, 'Index']

    avg = data.mean(numeric_only=True)
    data.loc['Average'] = avg
    data = data.fillna('-')

    cmaps = [cmap, f"{cmap}_r"] if reverse_cmap else [f"{cmap}_r", cmap]

    data_styled = data.style.background_gradient(subset=[('Accuracy', 'Train'), 
                                                         ('Accuracy', 'Validation'), 
                                                         ('Accuracy', 'Test')], axis=None, vmin=0, vmax=1, cmap=cmaps[0]) \
                        .background_gradient(subset=[('MAE', 'Train'), 
                                                     ('MAE', 'Validation'), 
                                                     ('MAE', 'Test')], vmin=0, vmax=1, axis=None, cmap=cmaps[1]) \
                    .format(precision=3) \
                    .set_table_styles(
                        [{'selector': 'th', 'props': [('text-align', 'center')]},  # Center align column names
                            {'selector': 'td', 'props': [('text-align', 'center')]},  # Center align values
                            {'selector': 'thead th', 'props': [('text-align', 'center')]},  # Center align indices
                            {'selector': 'td:nth-child(2)', 'props': [('border-left', '4px solid black')]},  # Add line between index and Test
                            {'selector': 'td:nth-child(4)', 'props': [('border-left', '4px solid black')]},  # Add line between val and train_mae
                            {'selector': 'td:nth-child(7)', 'props': [('border-left', '4px solid black')]},  # Add line between test_mae and train_accuracy
                            {'selector': 'td:nth-child(2)', 'props': [('width', '5em')]},
                            {'selector': 'td:nth-child(3)', 'props': [('width', '5em')]},
                            {'selector': 'td:nth-child(4)', 'props': [('width', '5em')]},
                            {'selector': 'td:nth-child(5)', 'props': [('width', '5em')]},
                            {'selector': 'td:nth-child(6)', 'props': [('width', '5em')]},
                            {'selector': 'td:nth-child(7)', 'props': [('width', '5em')]},
                            {'selector': 'td:nth-child(8)', 'props': [('width', '5em')]},
                            {'selector': 'td:nth-child(9)', 'props': [('width', '5em')]},
                            {'selector': 'thead tr:nth-child(1) th:nth-child(2)', 'props': [('border-left', '4px solid black')]},
                            {'selector': 'thead tr:nth-child(1) th:nth-child(3)', 'props': [('border-left', '4px solid black')]},
                            {'selector': 'thead tr:nth-child(1) th:nth-child(4)', 'props': [('border-left', '4px solid black')]},
                            {'selector': 'thead tr:nth-child(2) th:nth-child(2)', 'props': [('border-left', '4px solid black')]},
                            {'selector': 'thead tr:nth-child(2) th:nth-child(4)', 'props': [('border-left', '4px solid black')]},
                            {'selector': 'thead tr:nth-child(2) th:nth-child(7)', 'props': [('border-left', '4px solid black')]},
                            {'selector': f'tr:nth-child({data.shape[0]})', 'props': [('border-top', '4px solid black')]},
                            {'selector': f'tr:nth-child({data.shape[0]})', 'props': [('border-bottom', '4px solid black')]},
                            {'selector': f'tr:nth-child({data.shape[0]})', 'props': [('font-weight', 'bold')]}
                            ]  # Add line above the last horizontal row
                    ) \
                    .set_properties(**{'text-align': 'center'})  # Ensure all text is center aligned

    for i in range(maximum_tries):
        try:
            await dfi.export_async(obj=data_styled, filename=output_file_path, table_conversion='playwright')
            print(f"Try Number {i} Succeeded.")
            return
        except:
            print(f"Try Number {i} Failed.")


# if __name__ == "__main__":
#     asyncio.run(table_results(results_directory_path = '/home/mojtaba.kananisarche/Zh/outputs/2025-04-15|23:45:55|RNN_FACENET_5CV', 
#                               reverse_cmap=False, 
#                               cmap="coolwarm"))