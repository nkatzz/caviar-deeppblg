from typing import Iterator, Mapping
from deepproblog.query import Query
from problog.logic import Constant, Term
import torch
from deepproblog.dataset import Dataset

complex_event_mapping = {"nointeraction": 0, "meeting": 1, "moving": 2}


class CaviarVideos(Mapping[Term, torch.Tensor]):
    def __init__(self, videos: torch.Tensor):
        self.videos = videos

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)):
            yield self.videos[i]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i: tuple[Term, ...]):
        return self.videos[int(i[0])]


class CaviarDataset(Dataset):
    def __init__(
        self,
        complex_event_labels: torch.Tensor,
        is_train: bool = True,
        complex_event: str = "meeting",
    ):
        self.complex_event_labels = complex_event_labels
        self.subset_name = "train" if is_train else "test"
        self.complex_event = complex_event

    def __len__(self):
        return self.complex_event_labels.shape[0] * self.complex_event_labels.shape[1]

    def to_query(self, i: int) -> Query:
        video_id, timestep_number = (
            i // self.complex_event_labels.shape[1],
            i % self.complex_event_labels.shape[1],
        )

        return Query(
            Term(
                "holdsAt",
                Term("tensor", Term(self.subset_name, Constant(video_id))),
                Term(self.complex_event, Term("p1"), Term("p2")),
                Constant(timestep_number),
            ),
            p=(
                1
                if self.complex_event_labels[video_id][timestep_number].item()
                == complex_event_mapping[self.complex_event]
                else 0
            ),
        )
