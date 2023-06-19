import copy
from typing import Sequence
from deepproblog.query import Query
from problog.logic import Term, Constant
from .model import Model, Result
import re

class MarkovModel(Model):
    def solve(self, batch: Sequence[Query]) -> list[Result]:
        result = super().solve(batch)
        (body, probability) = list(result[0].result.items())[0]
        program = copy.deepcopy(self.program)
        
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
                'orientation',
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
        print('===============================')
        print(list(program))
        return result
