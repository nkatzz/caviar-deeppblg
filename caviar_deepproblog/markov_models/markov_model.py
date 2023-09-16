import copy
from deepproblog.query import Query
from problog.logic import Term, Constant
from deepproblog.model import Model, Result
from typing import (
    Collection,
    Optional,
    Sequence,
    Union,
)
from deepproblog.network import Network
import os
from deepproblog.embeddings import TermEmbedder
from time import time


class MarkovModel(Model):

    def __init__(self,
                 program_string: Union[str, os.PathLike],
                 networks: Collection[Network],
                 embeddings: Optional[TermEmbedder] = None,
                 load: bool = True):
        self.ac_total_time = 0  # Grounding + AC compilation + solving time
        self.deep_copy_time = 0
        super(MarkovModel, self).__init__(program_string, networks, embeddings=embeddings, load=load)

    def solve(self, batch: Sequence[Query]) -> list[Result]:
        result = super().solve(batch)
        self.ac_total_time += result[0].ground_time + result[0].compile_time + result[0].eval_time
        (body, probability) = list(result[0].result.items())[0]

        start = time()
        program = copy.deepcopy(self.program)
        end = time()
        self.deep_copy_time += end - start

        program.add_fact(
            Term(
                "previous_step",
                *body.args,
                p=Constant(
                    probability
                    if isinstance(probability, float)
                    else probability.item()
                ),
            )
        )

        program.add_fact(
            Term(
                "distance",
                body.args[0],
                *body.args[1].args,
                body.args[2],
                Constant(self.get_tensor(body.args[0])[int(body.args[2])][-1].item()),
            )
        )

        program.add_fact(
            Term(
                "orientation",
                body.args[0],
                *body.args[1].args,
                body.args[2],
                Constant(self.get_tensor(body.args[0])[int(body.args[2])][-2].item()),
            )
        )

        # TODO: The above line can be used to assign the spatial information
        # to the program. However currently all features are normalized which
        # means that it is hard to assign close and far based on the normalized
        # x1, y1, x2, y2 coordinates.
        if self.solver is None:
            raise RuntimeError(
                "no solver assigned. This will never happen. It is for the type checker"
            )
        self.solver.program = self.solver.engine.prepare(program)
        # print("===============================")
        # print(list(program))
        return result
