from collections import defaultdict
import torch
import pickle
import os

with open(
    os.path.join(os.getcwd(), "caviar_deepproblog/data/caviar_folds.pkl"), "rb"
) as data_file:
    caviar_folds = pickle.load(data_file)

import re

caviar_folds["fold1"]["train"][0]["atoms"]


def load_fold(
    fold_id: int, close_threshold_value: int = 25
) -> dict[str, dict[str, torch.Tensor]]:
    if fold_id not in range(1, 4):
        raise RuntimeError("There are only three folds with ids 1, 2, 3")

    caviar_folds["fold2"]["train"][0]["concat_tensor"][:, :, 10].min()

    fold_data = defaultdict(dict)

    for key in ("train", "test"):
        split_data = caviar_folds["fold{}".format(fold_id)][key]

        person_one_coords = torch.Tensor(
            [
                [
                    (int(x), int(y))
                    for x, y, _ in re.findall(
                        r"coords\(p1,(\d+),(\d+),(\d+)\)",
                        datapoint["atoms"],
                    )
                ]
                for datapoint in split_data
            ]
        )

        person_two_coords = torch.Tensor(
            [
                [
                    (int(x), int(y))
                    for x, y, _ in re.findall(
                        r"coords\(p2,(\d+),(\d+),(\d+)\)",
                        datapoint["atoms"],
                    )
                ]
                for datapoint in split_data
            ]
        )

        person_one_orientation = torch.Tensor(
            [
                [
                    int(x)
                    for x in re.findall(r"orientation\(p1,(\d+)", datapoint["atoms"])
                ]
                for datapoint in split_data
            ]
        )

        person_two_orientation = torch.Tensor(
            [
                [
                    int(x)
                    for x in re.findall(r"orientation\(p2,(\d+)", datapoint["atoms"])
                ]
                for datapoint in split_data
            ]
        )

        euclidean_distances = (
            ((person_one_coords - person_two_coords) ** 2).sum(-1).sqrt()
        ).unsqueeze(-1)

        orientation = (
            (person_one_orientation - person_two_orientation).abs()
        ).unsqueeze(-1)

        fold_input = torch.stack(
            [
                torch.cat(
                    [
                        split_data[example_id][key].squeeze(0)
                        for key in ["p1_tensor", "p2_tensor"]
                    ],
                    dim=1,
                )
                for example_id in range(len(split_data))
            ]
        )

        with_orientation_feature = torch.cat((fold_input, orientation), dim=-1)
        with_distance_feature = torch.cat(
            (with_orientation_feature, euclidean_distances), dim=-1
        )

        simple_event_labels = torch.stack(
            [
                torch.cat(
                    [
                        split_data[example_idx][key]
                        for key in ["p1_labels", "p2_labels"]
                    ],
                    dim=1,
                )
                for example_idx in range(len(split_data))
            ]
        )

        p1_features = torch.flatten(
            torch.stack(
                [
                    split_data[example_idx]["p1_tensor"]
                    for example_idx in range(len(split_data))
                ]
            ),
            end_dim=1,
        )

        p2_features = torch.flatten(
            torch.stack(
                [
                    split_data[example_idx]["p2_tensor"]
                    for example_idx in range(len(split_data))
                ]
            ),
            end_dim=1,
        )

        p1_simple_event_labels = torch.flatten(
            torch.stack(
                [
                    split_data[example_idx]["p1_labels"]
                    for example_idx in range(len(split_data))
                ]
            ),
            start_dim=1,
        )

        p2_simple_event_labels = torch.flatten(
            torch.stack(
                [
                    split_data[example_idx]["p2_labels"]
                    for example_idx in range(len(split_data))
                ]
            ),
            start_dim=1,
        )

        complex_event_labels = torch.stack(
            [
                split_data[example_id]["complex_labels"]
                for example_id in range(len(split_data))
            ]
        ).squeeze(-1)
        fold_data[key]["videos"] = with_distance_feature
        fold_data[key]["p1_features"] = p1_features
        fold_data[key]["p2_features"] = p2_features
        fold_data[key]["p1_simple_event_labels"] = p1_simple_event_labels
        fold_data[key]["p2_simple_event_labels"] = p2_simple_event_labels
        fold_data[key]["complex_event_labels"] = complex_event_labels
        fold_data[key]["simple_event_labels"] = simple_event_labels

    return fold_data
