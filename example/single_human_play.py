# pylint: disable=no-member

from time import perf_counter
import pygame
import gymnasium as gym

from eve_bench.aorticarch.arch_vmr94 import ArchVMR94, ObservationType

# import matplotlib.pyplot as plt

# env = ArchVMR94(init_visual=True, normalize_obs=False, obs_type=ObservationType.IMAGE)

env = gym.make("eve_bench/arch_vmr94", init_visual=True)

obs = env.observation_space.sample()

n_steps = 0
r_cum = 0.0

env.reset()

while True:
    start = perf_counter()
    trans = 0.0
    rot = 0.0
    pygame.event.get()
    keys_pressed = pygame.key.get_pressed()

    if keys_pressed[pygame.K_ESCAPE]:
        break
    if keys_pressed[pygame.K_UP]:
        trans += 50
    if keys_pressed[pygame.K_DOWN]:
        trans -= 50
    if keys_pressed[pygame.K_LEFT]:
        rot += 1 * 3.14
    if keys_pressed[pygame.K_RIGHT]:
        rot -= 1 * 3.14
    action = (trans, rot)
    obs, reward, terminal, trunc, info = env.step(action=action)
    image = env.render()
    # plt.imshow(image)
    # plt.show()
    n_steps += 1
    print(obs)

    if keys_pressed[pygame.K_RETURN] or terminal:
        env.reset()
        n_steps = 0

    print(f"FPS: {1/(perf_counter()-start)}")
env.close()
