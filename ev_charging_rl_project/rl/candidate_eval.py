import numpy as np
from typing import List
from helperstuff.simulatorstuff.candidates import Candidate

def score_candidates(
    candidates: List[Candidate],
    weight_time: float = 1.0,
    weight_cost: float = 1.0,
    weight_soc: float = 0.0
) -> List[float]:
    """
    Returns a score for each candidate based on weighted sum of:
    - detour time
    - charging cost
    - final state-of-charge after charging
    Lower time/cost → higher score
    """
    if not candidates:
        return []

    detour_times = np.array([c.detour_time_s for c in candidates])
    costs = np.array([c.cost for c in candidates])
    socs = np.array([c.final_soc for c in candidates])

    # Normalize
    if detour_times.ptp() > 0:
        detour_times = (detour_times.max() - detour_times) / detour_times.ptp()
    else:
        detour_times = np.ones_like(detour_times)

    if costs.ptp() > 0:
        costs = (costs.max() - costs) / costs.ptp()
    else:
        costs = np.ones_like(costs)

    if socs.ptp() > 0:
        socs = (socs - socs.min()) / socs.ptp()
    else:
        socs = np.zeros_like(socs)

    scores = (
        weight_time * detour_times +
        weight_cost * costs +
        weight_soc * socs
    )
    return scores.tolist()
