#!/usr/bin/env python3
"""
Copyright (C) 2025.
Python script for Berkeley DeepDrive Dataset Examination
Date: 26th March 2025
Authors:
- Gauresh Shirodkar (Modified by Gemini)
"""

# Standard imports
import os
import json
from collections import defaultdict

# Third party imports
import dash
import numpy as np
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import pandas as pd
from dash import dash_table


class DataAnalyzerBDD:
    """
    A class to analyze the Berkeley DeepDrive (BDD) dataset.
    """

    def __init__(self, training_set_path, validation_set_path):
        """
        Initializes the DataAnalyzerBDD with paths to the training and validation label files.

        Args:
            training_set_path (str): Path to the training labels JSON file.
            validation_set_path (str): Path to the validation labels JSON file.
        """
        self.training_set_path = training_set_path
        self.validation_set_path = validation_set_path
        self.training_data = self._load_annotation_file(self.training_set_path)
        self.validation_data = self._load_annotation_file(self.validation_set_path)
        self.training_insights = self._examine_dataset(self.training_data)
        self.validation_insights = self._examine_dataset(self.validation_data)
        (
            self.training_item_counts,
            self.training_item_sizes,
            self.training_item_locations,
            self.training_item_dimensions,
        ) = self._process_annotations(self.training_data)
        (
            self.validation_item_counts,
            self.validation_item_sizes,
            self.validation_item_locations,
            self.validation_item_dimensions,
        ) = self._process_annotations(self.validation_data)

    def _load_annotation_file(self, file_path):
        """
        Loads the JSON annotation file from the given path.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            list: A list of annotations loaded from the file, or an empty list if an error occurs.
        """
        try:
            with open(file_path, "r") as json_file:
                return json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error encountered while loading the file at {file_path}")
            return []

    def _examine_dataset(self, annotation_list):
        """
        Analyzes the dataset to extract information about categories, weather conditions,
        time of day, and object resolutions.

        Args:
            annotation_list (list): A list of annotation dictionaries.

        Returns:
            tuple: A tuple containing dictionaries for category counts, weather condition counts,
                   time of day counts, and a list of resolutions.
        """
        object_categories = defaultdict(int)
        sky_conditions = defaultdict(int)
        daytime = defaultdict(int)
        image_resolutions = []
        small_object_count = 0

        for record in annotation_list:
            for label_info in record.get("labels", []):
                object_categories[label_info["category"]] += 1

            sky_conditions[record.get("attributes", {}).get("weather", "unknown")] += 1
            daytime[record.get("attributes", {}).get("timeofday", "unknown")] += 1

            for obj_data in record["labels"]:
                if obj_data["category"] in [
                    "person",
                    "traffic light",
                    "truck",
                    "bus",
                    "bike",
                    "car",
                    "rider",
                    "traffic sign",
                ]:
                    box = obj_data.get("box2d", {})
                    box_width = box.get("x2", 0) - box.get("x1", 0)
                    box_height = box.get("y2", 0) - box.get("y1", 0)
                    if box_height and box_width:
                        image_resolutions.append((box_width, box_height))
                    if box_height < 20 and box_width < 20:
                        small_object_count += 1

        return object_categories, sky_conditions, daytime, image_resolutions

    def _process_annotations(self, annotation_list):
        """
        Processes the annotation data to count objects, their sizes, and their positions, and dimensions.

        Args:
            annotation_list (list): A list of annotation dictionaries.

        Returns:
            tuple: A tuple containing dictionaries for object counts, object sizes, and object positions.
        """
        item_occurrences = defaultdict(int)
        item_dimensions = defaultdict(list)
        item_placements = defaultdict(list)
        item_wh = defaultdict(lambda: {"width": [], "height": []})

        for record in annotation_list:
            for label_info in record.get("labels", []):
                item_class = label_info["category"]
                item_occurrences[item_class] += 1

                bounding_box = label_info.get("box2d", {})
                if bounding_box:
                    width_box = bounding_box["x2"] - bounding_box["x1"]
                    height_box = bounding_box["y2"] - bounding_box["y1"]
                    item_dimensions[item_class].append(width_box * height_box)
                    item_placements[item_class].append(
                        (
                            (bounding_box["x1"] + bounding_box["x2"]) / 2,
                            (bounding_box["y1"] + bounding_box["y2"]) / 2,
                        )
                    )
                    item_wh[item_class]["width"].append(width_box)
                    item_wh[item_class]["height"].append(height_box)
        return item_occurrences, item_dimensions, item_placements, item_wh

    def _calculate_percentiles(self, data):
        """Calculates specified percentiles for a list of numerical data."""
        if not data:
            return {}
        return {
            "1st Percentile": np.percentile(data, 1),
            "10th Percentile": np.percentile(data, 10),
            "25th Percentile": np.percentile(data, 25),
            "Median": np.median(data),
            "75th Percentile": np.percentile(data, 75),
            "90th Percentile": np.percentile(data, 90),
            "99th Percentile": np.percentile(data, 99),
        }


def main_analysis():
    """
    Main function to initialize and run the BDD dataset analysis dashboard.
    """
    TRAIN_ANNOTATION_PATH = r"D:\database\bdd_dataset\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"
    VAL_ANNOTATION_PATH = r"D:\database\bdd_dataset\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json"

    data_examiner = DataAnalyzerBDD(TRAIN_ANNOTATION_PATH, VAL_ANNOTATION_PATH)

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("Berkeley DeepDrive Dataset Insights"),
            dcc.RadioItems(
                id="dataset-selector",
                options=[
                    {"label": "Training Data", "value": "train"},
                    {"label": "Validation Data", "value": "val"},
                ],
                value="train",
                inline=True,
            ),
            dcc.Graph(id="object-category-distribution"),
            html.Div(
                style={"display": "flex"},
                children=[
                    dcc.Graph(id="weather-pie-chart", style={"width": "50%"}),
                    dcc.Graph(id="timeofday-pie-chart", style={"width": "50%"}),
                ],
            ),
            dcc.Graph(id="item-frequency-plot"),
            dcc.Graph(id="item-size-plot"),
            dcc.Graph(id="item-spatial-distribution"),
            html.Div(id="object-dimensions-tables"),  # Placeholder for the tables
        ]
    )

    @app.callback(
        [
            Output("object-category-distribution", "figure"),
            Output("weather-pie-chart", "figure"),
            Output("timeofday-pie-chart", "figure"),
            Output("item-frequency-plot", "figure"),
            Output("item-size-plot", "figure"),
            Output("item-spatial-distribution", "figure"),
            Output("object-dimensions-tables", "children"),
        ],  # New output for tables
        [Input("dataset-selector", "value")],
    )
    def update_visualizations(selected_data_set):
        if selected_data_set == "train":
            analysis_results = data_examiner.training_insights
            item_counts, item_sizes, item_locations, item_wh = (
                data_examiner.training_item_counts,
                data_examiner.training_item_sizes,
                data_examiner.training_item_locations,
                data_examiner.training_item_dimensions,
            )
        else:
            analysis_results = data_examiner.validation_insights
            item_counts, item_sizes, item_locations, item_wh = (
                data_examiner.validation_item_counts,
                data_examiner.validation_item_sizes,
                data_examiner.validation_item_locations,
                data_examiner.validation_item_dimensions,
            )

        object_categories, sky_conditions, daytime, image_resolutions = analysis_results

        category_figure = px.bar(
            x=list(object_categories.keys()),
            y=list(object_categories.values()),
            labels={"x": "Object Category", "y": "Number of Instances"},
            title="Distribution of Object Categories",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )

        weather_figure = px.pie(
            names=list(sky_conditions.keys()),
            values=list(sky_conditions.values()),
            title="Weather Conditions",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        timeofday_figure = px.pie(
            names=list(daytime.keys()),
            values=list(daytime.values()),
            title="Time of Day",
            color_discrete_sequence=px.colors.qualitative.Pastel1,
        )

        frequency_figure = px.bar(
            x=list(item_counts.keys()),
            y=list(item_counts.values()),
            labels={"x": "Detected Object", "y": "Frequency"},
            title="Frequency of Detected Objects",
            color_discrete_sequence=px.colors.qualitative.Dark2,
        )

        size_df = pd.DataFrame(
            [(k, v) for k, sizes_list in item_sizes.items() for v in sizes_list],
            columns=["Object Class", "Size"],
        )
        boxplot_figure = px.box(
            size_df,
            x="Object Class",
            y="Size",
            title="Distribution of Object Sizes (Log Scale)",
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            log_y=True,
        )

        spatial_figure = make_subplots(
            rows=min(5, (len(item_locations) + 1) // 2),
            cols=min(2, len(item_locations) if len(item_locations) > 1 else 1),
            subplot_titles=[
                f"Spatial Distribution of {obj}"
                for obj in sorted(item_locations.keys())
            ],
        )
        row_num, col_num = 1, 1
        color_palette = px.colors.qualitative.Vivid
        for i, obj_class in enumerate(sorted(item_locations.keys())):
            positions = item_locations[obj_class]
            if positions:
                x_coords, y_coords = zip(*positions)
                spatial_figure.add_trace(
                    px.scatter(
                        x=x_coords,
                        y=y_coords,
                        color_discrete_sequence=[color_palette[i % len(color_palette)]],
                        labels={"x": "X-Coordinate", "y": "Y-Coordinate"},
                        showlegend=False,
                    ).data[0],
                    row=row_num,
                    col=col_num,
                )
            else:
                spatial_figure.add_trace(
                    px.scatter(
                        x=[],
                        y=[],
                        labels={"x": "X-Coordinate", "y": "Y-Coordinate"},
                        showlegend=False,
                    ).data[0],
                    row=row_num,
                    col=col_num,
                )
            if col_num < 2:
                col_num += 1
            else:
                row_num += 1
                col_num = 1
            if row_num > 5:
                break

        spatial_figure.update_layout(
            title_text="Spatial Distribution of Detected Objects",
            height=1000,
            yaxis={"range": [750, 0]},
            grid={
                "rows": min(5, (len(item_locations) + 1) // 2),
                "columns": min(
                    2, len(item_locations) if len(item_locations) > 1 else 1
                ),
                "vertical_spacing": 0.1,
                "horizontal_spacing": 0.05,
            },
        )

        tables = []
        for category, dimensions in item_wh.items():
            width_stats = data_examiner._calculate_percentiles(dimensions["width"])
            height_stats = data_examiner._calculate_percentiles(dimensions["height"])

            width_table = dash_table.DataTable(
                id=f"width-table-{category}",
                columns=[
                    {"name": "Statistic", "id": "Statistic"},
                    {"name": "Value", "id": "Value"},
                ],
                data=[{"Statistic": k, "Value": v} for k, v in width_stats.items()],
                title=f"Width Statistics - {category}",
            )
            height_table = dash_table.DataTable(
                id=f"height-table-{category}",
                columns=[
                    {"name": "Statistic", "id": "Statistic"},
                    {"name": "Value", "id": "Value"},
                ],
                data=[{"Statistic": k, "Value": v} for k, v in height_stats.items()],
                title=f"Height Statistics - {category}",
            )
            tables.extend(
                [
                    html.H3(f"Category: {category}"),
                    html.H4("Width Statistics"),
                    width_table,
                    html.H4("Height Statistics"),
                    height_table,
                    html.Hr(),
                ]
            )

        return (
            category_figure,
            weather_figure,
            timeofday_figure,
            frequency_figure,
            boxplot_figure,
            spatial_figure,
            tables,
        )

    app.run(debug=True, port=8050)


if __name__ == "__main__":
    main_analysis()

"""
docker run --rm -p 8050:8050 -v "C:\Personal\Bosch_assignment\:/data" -e TRAIN_LABELS_PATH=/data/assignment_data_bdd_files/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json -e VAL_LABELS_PATH=/data/assignment_data_bdd_files/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json python-data-container

"""
