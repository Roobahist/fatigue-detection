import json
import os
from typing import Optional

import dataframe_image as dfi
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from Zh.src.utils.common import flatten
from Zh.src.utils.file_folder import create_folders


def table_results(
    results_directory_path: str,
    output_directory_path: Optional[str] = None,
    output_file_path_prefix: Optional[str] = "results",
    cmap: Optional[str] = "Blues",
    reverse_cmap: Optional[bool] = False,
    header_1_bg_color: Optional[str] = "silver",
    header_1_text_color: Optional[str] = "black",
    header_2_bg_color: Optional[str] = "darkgrey",
    header_2_text_color: Optional[str] = "black",
    maximum_tries: Optional[int] = 1,
    slice_or_video: Optional[str] = "slice",  # slice, video
    video_voting_method: Optional[str] = "average",  # majority, class_average, average
    height: Optional[int] = 2,
    width: Optional[int] = 5,
) -> None:

    KEYS_TO_COLLECT = ["split_info", "evaluation"]

    COLORS = [
        "Accent",
        "Blues",
        "BrBG",
        "BuGn",
        "BuPu",
        "CMRmap",
        "Dark2",
        "GnBu",
        "Greens",
        "Greys",
        "OrRd",
        "Oranges",
        "PRGn",
        "Paired",
        "Pastel1",
        "Pastel2",
        "PiYG",
        "PuBuGn",
        "PuBu",
        "PuOr",
        "PuRd",
        "Purples",
        "RdBu",
        "RdGy",
        "RdPu",
        "RdYlBu",
        "RdYlGn",
        "Reds",
        "Set1",
        "Set2",
        "Set3",
        "Spectral",
        "Wistia",
        "YlGnBu",
        "YlGn",
        "YlOrBr",
        "YlOrRd",
        "afmhot",
        "autumn",
        "binary",
        "bone",
        "brg",
        "bwr",
        "cividis",
        "cool",
        "coolwarm",
        "copper",
        "cubehelix",
        "flag",
        "gist_earth",
        "gist_gray",
        "gist_heat",
        "gist_ncar",
        "gist",
        "gist_rainbow",
        "gist_stern",
        "gist_yarg",
        "gnuplot2",
        "gnuplot",
        "gray",
        "hot",
        "hsv",
        "inferno",
        "jet",
        "magma",
        "nipy_spectral",
        "ocean",
        "pink",
        "plasma",
        "prism",
        "rainbow",
        "seismic",
        "spring",
        "summer",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
        "terrain",
        "turbo",
        "twilight",
        "twilight_shifted",
        "viridis",
        "winter",
    ]

    output_directory_path = output_directory_path or results_directory_path
    create_folders(output_directory_path)
    output_file_name = None

    i = 0
    while output_file_name is None:
        i += 1
        if not os.path.exists(
            os.path.join(output_directory_path, f"{output_file_path_prefix}_{i}.png")
        ):
            output_file_name = f"{output_file_path_prefix}_{i}.png"

    output_file_path = os.path.join(output_directory_path, output_file_name)

    data = []
    for file_name in os.listdir(results_directory_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(results_directory_path, file_name)
            with open(file=file_path, mode="r") as json_file:
                j = json.load(json_file)
                try:
                    data.append({k: v for k, v in j.items() if k in KEYS_TO_COLLECT})
                except KeyError as e:
                    print("Unable")
                    continue

    data = pd.DataFrame(data=data).dropna(how="all")

    folded_by_column = list(data["split_info"].iloc[0]["test"].keys())[0]

    data.index = data["split_info"].apply(lambda x: x["index"]).astype(int).tolist()
    data = data.sort_index()

    data["split_info_train"] = data["split_info"].apply(
        lambda x: x["train"][folded_by_column]
    )
    data["split_info_val"] = data["split_info"].apply(
        lambda x: x["val"][folded_by_column]
    )
    data["split_info_test"] = data["split_info"].apply(
        lambda x: x["test"][folded_by_column]
    )

    data["evaluation_train"] = data["evaluation"].apply(
        lambda x: x["train"]["metrics"][slice_or_video]
    )
    data["evaluation_val"] = data["evaluation"].apply(
        lambda x: x["val"]["metrics"][slice_or_video]
    )
    data["evaluation_test"] = data["evaluation"].apply(
        lambda x: x["test"]["metrics"][slice_or_video]
    )

    if slice_or_video == "video":
        data["evaluation_train"] = data["evaluation_train"].apply(
            lambda x: x[video_voting_method]
        )
        data["evaluation_val"] = data["evaluation_val"].apply(
            lambda x: x[video_voting_method]
        )
        data["evaluation_test"] = data["evaluation_test"].apply(
            lambda x: x[video_voting_method]
        )

    data["evaluation_train_accuracy"] = data["evaluation_train"].apply(
        lambda x: x["accuracy"]
    )
    data["evaluation_val_accuracy"] = data["evaluation_val"].apply(
        lambda x: x["accuracy"]
    )
    data["evaluation_test_accuracy"] = data["evaluation_test"].apply(
        lambda x: x["accuracy"]
    )

    data["evaluation_train_mae"] = data["evaluation_train"].apply(lambda x: x["mae"])
    data["evaluation_val_mae"] = data["evaluation_val"].apply(lambda x: x["mae"])
    data["evaluation_test_mae"] = data["evaluation_test"].apply(lambda x: x["mae"])

    printable_data = data[
        [
            "split_info_test",
            "split_info_val",
            "evaluation_train_mae",
            "evaluation_val_mae",
            "evaluation_test_mae",
            "evaluation_train_accuracy",
            "evaluation_val_accuracy",
            "evaluation_test_accuracy",
        ]
    ].copy()

    multi_index = pd.MultiIndex.from_tuples(
        [
            (f"Split ({folded_by_column})", "Test"),
            (f"Split ({folded_by_column})", "Validation"),
            ("MAE", "Train"),
            ("MAE", "Validation"),
            ("MAE", "Test"),
            ("Accuracy", "Train"),
            ("Accuracy", "Validation"),
            ("Accuracy", "Test"),
        ]
    )
    printable_data.columns = multi_index
    printable_data.columns.names = [None, "Index"]

    avg = printable_data.mean(numeric_only=True)
    printable_data.loc["Average"] = avg
    printable_data = printable_data.fillna("-")

    cmaps = [f"{cmap}_r", cmap] if reverse_cmap else [cmap, f"{cmap}_r"]

    data_styled = (
        printable_data.style.background_gradient(
            subset=[
                ("Accuracy", "Train"),
                ("Accuracy", "Validation"),
                ("Accuracy", "Test"),
            ],
            axis=None,
            vmin=0,
            vmax=1,
            cmap=cmaps[0],
        )
        .format(precision=3)
        .background_gradient(
            subset=[("MAE", "Train"), ("MAE", "Validation"), ("MAE", "Test")],
            vmin=0,
            vmax=1,
            axis=None,
            cmap=cmaps[1],
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("height", f"{height}em"),
                        ("width", f"{width}em"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("height", f"{height}em"),
                        ("width", f"{width}em"),
                    ],
                },
                {
                    "selector": "thead tr:nth-child(1) th:nth-child(1)",
                    "props": [("background-color", "white")],
                },
                {
                    "selector": "thead tr:nth-child(1)",
                    "props": [
                        ("background-color", header_1_bg_color),
                        ("color", header_1_text_color),
                    ],
                },
                {
                    "selector": "thead tr:nth-child(2)",
                    "props": [
                        ("background-color", header_2_bg_color),
                        ("color", header_2_text_color),
                    ],
                },
                # header row 1 vertical line
                {
                    "selector": "thead tr:nth-child(1) th:nth-child(1)",
                    "props": [("border-right", "15px solid white")],
                },
                {
                    "selector": "thead tr:nth-child(1) th:nth-child(2)",
                    "props": [("border-right", "15px solid white")],
                },
                {
                    "selector": "thead tr:nth-child(1) th:nth-child(3)",
                    "props": [("border-right", "15px solid white")],
                },
                # header row 2 vertical line
                {
                    "selector": "thead tr:nth-child(2) th:nth-child(1)",
                    "props": [("border-right", "15px solid white")],
                },
                {
                    "selector": "thead tr:nth-child(2) th:nth-child(3)",
                    "props": [("border-right", "15px solid white")],
                },
                {
                    "selector": "thead tr:nth-child(2) th:nth-child(6)",
                    "props": [("border-right", "15px solid white")],
                },
                # body vertical line
                {
                    "selector": "tbody th:nth-child(1)",
                    "props": [("border-right", "15px solid white")],
                },
                {
                    "selector": "tbody td:nth-child(3)",
                    "props": [("border-right", "15px solid white")],
                },
                {
                    "selector": "tbody td:nth-child(6)",
                    "props": [("border-right", "15px solid white")],
                },
                # Bottom line
                {
                    "selector": f"tbody tr:nth-child({printable_data.shape[0]})",
                    "props": [("border-top", "3px solid SpringGreen")],
                },
                {
                    "selector": f"tbody tr:nth-child({printable_data.shape[0]})",
                    "props": [("border-bottom", "3px solid SpringGreen")],
                },
                {
                    "selector": f"tbody tr:nth-child({printable_data.shape[0]})",
                    "props": [("font-weight", "bold")],
                },
            ]
        )
    )

    for attempt in range(maximum_tries):
        try:
            data_styled.export_png(
                filename=output_file_path, table_conversion="selenium"
            )
            print(f"Try Number {attempt} Succeeded.")
            break
        except:
            print(f"Try Number {attempt} Failed.")

    return data_styled


def confusion_matrix_results(
    results_directory_path: str,
    unique_classes: list[float],
    output_directory_path: Optional[str] = None,
    output_file_name_prefix: Optional[str] = "results",
    index: Optional[int] = -1,  # positive for any index and negative for all
    split: Optional[str] = "all",  # train, val, test, all
    cmap: Optional[str] = "Greys",
    maximum_tries: Optional[int] = 5,
    corner_text_color: Optional[str] = "white",
    corner_bg_color: Optional[str] = "IndianRed",
    pred_true_text_color: Optional[str] = "black",
    pred_true_bg_color: Optional[str] = "silver",
    labels_bg_color: Optional[str] = "beige",
    labels_text_color: Optional[str] = "black",
    height: int = 5,
    width: Optional[int] = None,
):

    width = width or height

    thresholds = []
    unique_classes = sorted(unique_classes)
    for i in range(len(unique_classes) - 1):
        thresholds.append((unique_classes[i] + unique_classes[i + 1]) / 2)

    KEYS_TO_COLLECT = [
        "index",
        "test",
        "val",
        "y_train",
        "y_train_pred",
        "y_val",
        "y_val_pred",
        "y_test",
        "y_test_pred",
    ]

    output_directory_path = output_directory_path or results_directory_path
    create_folders(output_directory_path)
    output_file_name = None
    i = 0
    while output_file_name is None:
        i += 1
        if not os.path.exists(
            os.path.join(output_directory_path, f"{output_file_name_prefix}_{i}.png")
        ):
            output_file_name = f"{output_file_name_prefix}_{i}.png"
    output_file_path = os.path.join(output_directory_path, output_file_name)

    data = []
    for file_name in os.listdir(results_directory_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(results_directory_path, file_name)
            with open(file=file_path, mode="r") as json_file:
                j = json.load(json_file)
                try:
                    data.append({k: v for k, v in j.items() if k in KEYS_TO_COLLECT})
                except KeyError as e:
                    print("Unable")
                    continue

    data = pd.DataFrame(data=data).dropna(how="all")
    try:
        data.index = data["index"].astype(int).values
        data = data.sort_index()
        data = data.drop("index", axis=1)
    except:
        print("No Index")
        pass

    folded_by_column = list(data["test"][0].keys())[0]

    def style(confusion_matrix_df):

        confusion_matrix_df.columns = pd.MultiIndex.from_product(
            iterables=[["Predicted"], [str(round(float(i), 1)) for i in unique_classes]]
        )
        confusion_matrix_df.index = pd.MultiIndex.from_product(
            iterables=[["True"], [str(round(float(i), 1)) for i in unique_classes]]
        )

        styled = confusion_matrix_df.style.background_gradient(
            axis=1, cmap=cmap
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("height", f"{height}em"),
                        ("width", f"{width}em"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("height", f"{height}em"),
                        ("width", f"{width}em"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "thead tr:nth-child(1) th:nth-child(1)",
                    "props": [
                        ("background-color", corner_bg_color),
                        ("color", corner_text_color),
                    ],
                },
                {
                    "selector": "th:nth-child(2)",
                    "props": [("border-right", "1px solid black")],
                },
                {
                    "selector": "td:nth-child(2)",
                    "props": [("border-left", "1px solid black")],
                },
                {
                    "selector": "thead tr:nth-child(1) th:nth-child(3)",
                    "props": [
                        ("background-color", pred_true_bg_color),
                        ("color", pred_true_text_color),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(1) th:nth-child(1)",
                    "props": [
                        ("background-color", pred_true_bg_color),
                        ("color", pred_true_text_color),
                    ],
                },
                {
                    "selector": "thead tr:nth-child(2) th:nth-child(3)",
                    "props": [
                        ("background-color", labels_bg_color),
                        ("color", labels_text_color),
                    ],
                },
                {
                    "selector": "thead tr:nth-child(2) th:nth-child(4)",
                    "props": [
                        ("background-color", labels_bg_color),
                        ("color", labels_text_color),
                    ],
                },
                {
                    "selector": "thead tr:nth-child(2) th:nth-child(5)",
                    "props": [
                        ("background-color", labels_bg_color),
                        ("color", labels_text_color),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(1) th:nth-child(2)",
                    "props": [
                        ("background-color", labels_bg_color),
                        ("color", labels_text_color),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(2) th:nth-child(1)",
                    "props": [
                        ("background-color", labels_bg_color),
                        ("color", labels_text_color),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(3) th:nth-child(1)",
                    "props": [
                        ("background-color", labels_bg_color),
                        ("color", labels_text_color),
                    ],
                },
            ]
        )
        return styled

    styled_dfs = {}
    if index > 0:
        split = ["train", "test", "val"] if split == "all" else [split]
        for s in split:
            y_true = [str(float(yt)) for yt in data[f"y_{s}"][index]]
            y_pred = [
                str(float(unique_classes[np.digitize(x, thresholds, right=False)]))
                for x in data[f"y_{s}_pred"][index]
            ]
            confusion_matrix_df = pd.DataFrame(
                data=confusion_matrix(y_true=y_true, y_pred=y_pred)
            )
            styled_df = style(confusion_matrix_df)
            styled_dfs[f"{s}"] = styled_df

    else:
        index = "combined"
        split = ["train", "test", "val"] if split == "all" else [split]
        for s in split:
            y_true = [str(float(yt)) for yt in flatten(list(data[f"y_{s}"]))]
            y_pred = [
                str(float(yp))
                for yp in flatten(
                    list(
                        data[f"y_{s}_pred"].apply(
                            lambda x: [
                                unique_classes[np.digitize(i, thresholds, right=False)]
                                for i in x
                            ]
                        )
                    )
                )
            ]
            confusion_matrix_df = pd.DataFrame(
                data=confusion_matrix(y_true=y_true, y_pred=y_pred)
            )
            styled_df = style(confusion_matrix_df)
            styled_dfs[f"{s}"] = styled_df

    for s, styled_df in styled_dfs.items():
        print(f"split - {s}")
        for attempt in range(maximum_tries):
            output_file_path = os.path.join(
                output_directory_path, f"{output_file_name_prefix}_{s}_{index}.png"
            )
            try:
                styled_df.export_png(
                    filename=output_file_path,
                    table_conversion="selenium",
                    corner=f"{str(index).capitalize()}<br><br>{s.capitalize() if s!='val' else 'Validation'}",
                )
                print(f"Try Number {attempt} Succeeded.")
                break
            except:
                print(f"Try Number {attempt} Failed.")

    return styled_dfs


# if __name__ == "__main__":
#     table_results(results_directory_path = '/home/mojtaba.kananisarche/Zh/outputs/2025-04-15|23:45:55|RNN_FACENET_5CV',
#                               reverse_cmap=False,
#                               cmap="coolwarm")
# data =table_results(
# results_directory_path  = '/home/mojtaba.kananisarche/Zh/outputs/2025-04-20|06:42:10|RNN_FACENET_5CV',
# output_directory_path   = None,
# output_file_path_prefix = 'results',
# cmap                    = "Blues",
# reverse_cmap            = False,
# header_1_bg_color       = 'silver',
# header_1_text_color     = 'black',
# header_2_bg_color       = 'darkgrey',
# header_2_text_color     = 'black',
# maximum_tries           = 1,
# slice_or_video          = 'slice', #slice, video
# video_voting_method     = 'average', #majority, class_average, average
# height                  = 2,
# width                   = 5,
# )