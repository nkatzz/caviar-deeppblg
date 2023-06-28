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
        if self.get_tensor(body.args[0])[int(body.args[2])][-1].item() <= 25:
            program.add_fact(
                Term(
                    "is_close",
                    body.args[0],
                    *body.args[1].args,
                    body.args[2],
                    Constant(25),
                )
            )
            program.add_fact(
                Term(
                    "is_close",
                    body.args[0],
                    *body.args[1].args,
                    body.args[2],
                    Constant(34),
                )
            )
        elif self.get_tensor(body.args[0])[int(body.args[2])][-1].item() <= 34:
            program.add_fact(
                Term(
                    "is_close",
                    body.args[0],
                    *body.args[1].args,
                    body.args[2],
                    Constant(34),
                )
            )
            program.add_fact(
                Term(
                    "far",
                    body.args[0],
                    *body.args[1].args,
                    body.args[2],
                    Constant(25),
                )
            )
        else:
            program.add_fact(
                Term(
                    "far",
                    body.args[0],
                    *body.args[1].args,
                    body.args[2],
                    Constant(25),
                )
            )
            program.add_fact(
                Term(
                    "far",
                    body.args[0],
                    *body.args[1].args,
                    body.args[2],
                    Constant(34),
                )
            )

        if self.get_tensor(body.args[0])[int(body.args[2])][-2].item() <= 45:
            program.add_fact(
                Term(
                    'orientation',
                    body.args[0],
                    *body.args[1].args,
                    body.args[2],
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
        #print('===============================')
        #print(list(program))
        return result
