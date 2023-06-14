from collections import defaultdict
import torch
import pickle

with open(
    "/home/whatever/programms/caviar-deepproblog/caviar_deepproblog/data/caviar_folds.pkl",
    "rb",
) as data_file:
    caviar_folds = pickle.load(data_file)


def load_fold(fold_id: int) -> dict[str, dict[str, torch.Tensor]]:
    if fold_id not in range(1, 4):
        raise RuntimeError("There are only three folds with ids 1, 2, 3")

    caviar_folds["fold2"]["train"][0]["concat_tensor"][:, :, 10].min()

    fold_data = defaultdict(dict)

    for key in ("train", "test"):
        split_data = caviar_folds["fold{}".format(fold_id)][key]
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
        complex_events_labels = torch.stack(
            [
                split_data[example_id]["complex_labels"]
                for example_id in range(len(split_data))
            ]
        ).squeeze(-1)
        fold_data[key]["videos"] = fold_input
        fold_data[key]["labels"] = complex_events_labels

    return fold_data
