import math
from typing import Optional

import torch

from deepproblog.optimizer import Optimizer
from deepproblog.semiring.graph_semiring import GraphSemiring, Result
from problog.logic import Term


class BoundEntropySemiring(GraphSemiring):
    @staticmethod
    def cross_entropy(
        result: Result,
        target: float,
        weight: float,
        q: Optional[Term] = None,
        eps: float = 1e-12,
    ) -> float:
        result = result.result
        if len(result) == 0:
            print("No results found for {}".format(q))
            return 0
        if q is None:
            if len(result) == 1:
                q, p = next(iter(result.items()))
            else:
                raise ValueError(
                    "q is None and number of results is {}".format(len(result))
                )
        else:
            p = result[q]
        if type(p) is float:
            loss = (
                -(target * math.log(p + eps) + (1.0 - target) * math.log(1.0 - p + eps))
                * weight
            )
        else:
            if target == 1.0:
                loss = -torch.log(p + eps) * weight
            elif target == 0.0:
                loss = -torch.log(1.0 - p + eps) * weight
            else:
                loss = (
                    -(
                        target * torch.log(p + eps)
                        + (1.0 - target) * torch.log(1.0 - p + eps)
                    )
                    * weight
                )
            loss.backward(retain_graph=True)
        return float(loss)