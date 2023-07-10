from typing import Union

from ..synthesize import GRNSimulator
from ..trajectories import Trajectory


def _lookup_trajectory(
    model: GRNSimulator,
    trajectory: Union[int, str]
) -> Trajectory:

    try:
        # Check for integer indexing
        trajectory = model._trajectories[trajectory]
    except (IndexError, TypeError):
        # If that doesnt work check names
        trajectory = {
            t.name: t
            for t in model._trajectories
        }.get(trajectory, None)

    if trajectory is None:
        raise KeyError(
            f"Can't find trajectory {trajectory} in this simulation"
        )

    return trajectory
