from collections import defaultdict, Counter
import torch
import pickle
import re


with open(
    "/home/whatever/programms/caviar-deepproblog/caviar_deepproblog/data/caviar_folds.pkl",
    # "/home/yuzer/ncsr/dpl-benchmark/caviar-deepproblog/caviar_deepproblog/data/caviar_folds.pkl",
    "rb",
) as data_file:
    caviar_folds = pickle.load(data_file)


def load_fold(
    fold_id: int, close_threshold_value: int = 25
) -> dict[str, dict[str, torch.Tensor]]:
    if fold_id not in range(1, 4):
        raise RuntimeError("There are only three folds with ids 1, 2, 3")

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

        euclidean_distances = (
            ((person_one_coords - person_two_coords) ** 2).sum(-1).sqrt()
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

        with_distance_feature = torch.cat((fold_input, euclidean_distances), dim=-1)

        complex_events_labels = torch.stack(
            [
                split_data[example_id]["complex_labels"]
                for example_id in range(len(split_data))
            ]
        ).squeeze(-1)
        fold_data[key]["videos"] = with_distance_feature
        fold_data[key]["labels"] = complex_events_labels

    return fold_data
