from ragas_jev.schemas import Decision
from ragas_jev.scoring.uncertainty import binary_confidence


def noul_decision(unit_id: str, p: float) -> Decision:
    conf = binary_confidence(p)
    return Decision(
        unit_id=unit_id,
        primitive="noul",
        question_id="test.v1",
        distribution={"true": p, "false": 1 - p},
        p=p,
        confidence=conf,
        uncertainty=1 - conf,
        entropy=0.0,
        jev_model="test",
        jev_p=p,
    )


def noul_decisions(prefix: str, ps: list[float]) -> list[Decision]:
    return [noul_decision(f"{prefix}_{i}", p) for i, p in enumerate(ps)]
