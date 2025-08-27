from env import CustomCarEnv

# if __name__ == "__main__":
#     env = CustomCarEnv()
#     obs = env.reset()
#     for _ in range(1000):
#         action = env.action_space.sample()  # Random action
#         obs, fr, done, info = env.step(action)
#         print(f"Observation: {obs}, Reward: {fr}, Done: {done}")
#         env.render()
#         if done:
#             obs = env.reset()
#     env.close()

if __name__ == "__main__":
    env = CustomCarEnv()
    env.reset()
    while True:
        env.render()
