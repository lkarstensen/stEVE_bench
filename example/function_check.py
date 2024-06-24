# pylint: disable=no-member

from time import perf_counter
import numpy as np

from eve_bench import ArchVariety
from eve.visualisation import SofaPygame

intervention = ArchVariety()
visu = SofaPygame(intervention)


n_steps = 0
r_cum = 0.0


for _ in range(3):
    intervention.reset()
    visu.reset()
    for _ in range(25):
        start = perf_counter()

        action = np.array([40.0, 0.2])
        intervention.step(action=action)
        visu.render()

        n_steps += 1

        print(f"FPS: {1/(perf_counter()-start)}")

intervention.close()
visu.close()
