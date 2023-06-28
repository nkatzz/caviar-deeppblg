import copy
from typing import Sequence
from deepproblog.query import Query
from problog.logic import Term, Constant
from deepproblog.model import Model, Result


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

        if self.solver is None:
            raise RuntimeError(
                "no solver assigned. This will never happen. It is for the type checker"
            )
        self.solver.program = self.solver.engine.prepare(program)
        # print(list(program))
        return result
