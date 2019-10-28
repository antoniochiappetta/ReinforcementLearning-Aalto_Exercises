import numpy as np
from time import sleep
from sailing import SailingGridworld

# Set up the environment
env = SailingGridworld(rock_penalty=-2, discount_factor=0.9, threshold=pow(10, -4))
value_est = np.zeros((env.w, env.h))
policy = np.zeros((env.w, env.h))
env.draw_values(value_est)

if __name__ == "__main__":

    # Reset the environment
    state = env.reset()

    episodes = 1000
    iterations = 100
    iterations_convergence = 0
    converged = False

    while not converged:
    # for _ in range(iterations):

        iterations_convergence += 1
        env.clear_text()

        # Compute state values and the policy
        value_est, policy, converged = env.state_value_iteration(value_est, policy)

        # Show the values and the policy
        env.draw_values(value_est)
        env.draw_actions(policy)
        # env.render()
        # sleep(1)

    print("The number of iterations performed before convergence was:")
    print(iterations_convergence)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run multiples episodes
    episode_counter = 0
    step_counter = 0
    discounted_return_dist = np.zeros(episodes)

    while episode_counter < episodes:

        # Select an optimal action
        action = policy[env.state]

        # Step the environment
        state, reward, done, _ = env.step(action)

        # Discounted return
        discounted_return_dist[episode_counter] += pow(env.discount_factor, step_counter) * reward
        step_counter += 1
        if done:
            print("Episode")
            print(episode_counter)
            print("completed\n")
            episode_counter += 1
            step_counter = 0
            if episode_counter < episodes:
                # Reset the environment
                state = env.reset()

    # Compute mean and sd of the discounted return of the initial state
    print("Mean of discounted return of the initial state")
    print(np.mean(discounted_return_dist))
    print("Standard deviation of discounted return of the initial state")
    print(np.std(discounted_return_dist))

    # Render the final outcome
    env.render()
