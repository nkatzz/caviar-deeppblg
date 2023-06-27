import torch
from typing import Iterator, Mapping
from problog.logic import Constant, Term
from deepproblog.query import Query
from deepproblog.dataset import Dataset
from caviar_deepproblog.data.caviar_vision_data import CaviarVisionDataset
from sklearn.model_selection import train_test_split


complex_event_mapping = {"nointeraction": 0, "meeting": 1, "moving": 2}


class CaviarImagePairs(Mapping[Term, torch.Tensor]):
    def __init__(self, videos: torch.Tensor):
        self.videos = videos

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)):
            yield self.videos[i]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i: tuple[Term, ...]):
        return self.videos[int(i[0])]


class CaviarVisionDPLDataset(Dataset):
    def __init__(self, subset, complex_event_labels, complex_event):
        self.complex_event_labels = complex_event_labels
        self.subset = subset
        self.complex_event = complex_event
        self.window_size = len(complex_event_labels[0])

    def __len__(self):
        return len(self.complex_event_labels) * self.window_size

    def to_query(self, idx: int) -> Query:
        video_id, timestep_number = (
            idx // self.window_size,
            idx % self.window_size,
        )

        return Query(
            Term(
                "holdsAt",
                Term("tensor", Term(self.subset, Constant(video_id))),
                Term(self.complex_event, Term("p1"), Term("p2")),
                Constant(timestep_number),
            ),
            p=(
                1
                if self.complex_event_labels[video_id][timestep_number]
                == complex_event_mapping[self.complex_event]
                else 0
            ),
        )
