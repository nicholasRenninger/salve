import jax.numpy as jnp
from jax import grad, jit, vmap, random


@jit
def U_attractive(q, q_goal, gain: float) -> float:
    return 0.5 * gain * jnp.linalg.norm(q_goal - q) ** 2


@jit
def U_repulsive(q, q_obstacles, dist_to_feel_repulsion, gain, obstacle_radius) -> float:
    if q_obstacles.ndim == 3:
        _, num_obs, obs_dim = q_obstacles.shape
    elif q_obstacles.ndim == 2:
        num_obs, obs_dim = q_obstacles.shape
    elif q_obstacles.ndim == 1:
        num_obs = obs_dim = len(q_obstacles)
    else:
        raise ValueError(f"ndim={q_obstacles.ndim} of q_obstacles not supported")

    if num_obs == 0 or obs_dim == 0:
        return 0.0

    q_vec = jnp.tile(q, (len(q_obstacles), 1))

    def distance(q_curr, q_obs):
        distances = jnp.linalg.norm(q_obs - q_curr, axis=0)
        return jnp.where(distances > obstacle_radius, distances, 0), distances

    def distance_field(q_curr, q_obs):
        distances, distances_to_obs_centers = distance(q_curr, q_obs)
        distance_field = (1 / distances) - (1 / dist_to_feel_repulsion)
        new_distance_field = jnp.where(
            (distances_to_obs_centers - (dist_to_feel_repulsion)) <= 0,
            distance_field,
            0,
        )

        return new_distance_field

    dist_vec = vmap(distance_field, in_axes=0, out_axes=0)

    return jnp.sum(0.5 * gain * dist_vec(q_vec, q_obstacles) ** 2, axis=0)


@jit
def U(
    q,
    q_goal,
    q_obstacles,
    dist_to_feel_repulsion=0.2,
    obstacle_radius=0.1,
    gain_attractive=10.0,
    gain_repulsive=1e0,
):

    if dist_to_feel_repulsion is None:
        dist_to_feel_repulsion = obstacle_radius

    return U_attractive(q=q, q_goal=q_goal, gain=gain_attractive) + U_repulsive(
        q=q,
        q_obstacles=q_obstacles,
        obstacle_radius=obstacle_radius,
        dist_to_feel_repulsion=dist_to_feel_repulsion,
        gain=gain_repulsive,
    )


grad_U = jit(grad(U))


def compute_safe_path(
    q_start,
    q_goal,
    q_obstacles,
    grad_U=grad_U,
    grad_desc_rate=1e-2,
    zero_grad_tol=1e-8,
    random_jitter_mag_cov=1e-4,
    N=1000,
):

    key = random.PRNGKey(0)
    q = q_start
    q_history = [q]
    iters = jnp.array(range(0, N))

    i = 0
    for i in iters:

        delta_q = grad_U(q, q_goal, q_obstacles)
        cov = jnp.linalg.norm(delta_q) * random_jitter_mag_cov
        delta_q_jitter = random.normal(key, shape=delta_q.shape) * cov
        # q_mag = jnp.diag(jnp.linalg.norm(delta_q, axis=0))
        # delta_q_jitter = random.multivariate_normal(
        #     key, mean=jnp.zeros(len(delta_q)), cov=q_mag / 4, shape=delta_q.shape
        # )

        q -= grad_desc_rate * (delta_q + delta_q_jitter)

        if jnp.linalg.norm(grad_U(q, q_goal, q_obstacles)) < zero_grad_tol:
            break

        q_history.append(q)
        i += 1

    return q_history


def compute(f, q_stacked_by_dim, q_goal, q_obstacles):
    f_vectorized = vmap(f, in_axes=(0, 0, 0), out_axes=0)
    n_evals, ndims = q_stacked_by_dim.shape

    return f_vectorized(
        q_stacked_by_dim,
        jnp.tile(q_goal, (n_evals, 1)),
        jnp.tile(q_obstacles, (n_evals, 1, 1)),
    )


def main():

    import numpy as np

    # q_start = jnp.array([1.0, 1.0])
    q_goal = jnp.array([0.0, 0.0])
    q_obstacles = jnp.array([[0.5, 0.5], [0.5, 0.1], [0.1, 0.5]])

    # q_history = compute_safe_path(
    #     q_start=q_start,
    #     q_goal=q_goal,
    #     q_obstacles=q_obstacles,
    #     # q_obstacles=jnp.array([]),
    #     grad_desc_rate=1e-3,
    #     zero_grad_tol=1e-4,
    #     N=1000,
    # )

    # print(q_history)

    grid_resolution = 50
    q0, q1 = np.mgrid[-1 : 1 : 1 / (grid_resolution), -1 : 1 : 1 / grid_resolution]
    q_vstacked = np.vstack(list(zip(q0.ravel(), q1.ravel())))

    U_values = compute(U, q_vstacked, q_goal, q_obstacles)
    print(U_values)


if __name__ == "__main__":
    main()
