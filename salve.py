import jax.numpy as jnp
import matplotlib.pyplot as plt

from enum import IntEnum
from typing import Callable, List, Tuple, Union
from dataclasses import dataclass
from functools import partial

from tensorflow import keras

from jax import grad, jit, vmap, random


# have to use ints or jax has a mental breakdown
class DistType(IntEnum):
    cosine = 0
    norm = 1


U_static_argnames = (
    "dist_to_feel_repulsion",
    "obstacle_radius",
    "gain_attractive",
    "gain_repulsive",
    "dist_type",
    "norm_ord",
)


@partial(jit, static_argnames=("dist_type", "norm_ord"))
def distance(
    q1: jnp.array,
    q2: jnp.array,
    dist_type: DistType = DistType.cosine,
    norm_ord: int = None,
) -> float:
    """scalar distance between two objects

    accepts vector, matrix/tensor objects.

    Args:
        q1: matrix/tensor or vector of object coordinates "start"
        q2: matrix/tensor or vector of object coordinates. "destination"
        dist_type: Type of distance metrics:
            "cosine": computes cosine distance (1 - cosine_similarity). If q are tensors, they're
                first flattened into vectors for angle computation.
                Distance ranges from:
                    min: 0
                    max: 2
            "norm": computes norm-based distance. Follow `numpy.linalg.norm()`'s guide for how to
                set `axis` and `ord` for tensor and vector `q`s.
                Distance ranges from:
                    min: 0
                    max: 2
        norm_ord: Order of the norm. inf means jax/numpy's inf object.
            ONLY USED IF dist_type = "norm".
            The default is None.
            follows numpy.linalg.norm():
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    Returns:
        scalar distance between q1 and q2
    """

    if dist_type == DistType.norm:
        norm_axis, _ = get_axis_and_tiling(q1)
        return jnp.linalg.norm(q2 - q1, axis=norm_axis, ord=norm_ord)

    elif dist_type == DistType.cosine:
        q1_norm = jnp.linalg.norm(q1.flatten(), axis=0, ord=norm_ord)
        q2_norm = jnp.linalg.norm(q2.flatten(), axis=0, ord=norm_ord)

        return 1 - jnp.dot(q1.flatten(), q2.flatten()) / (q1_norm * q2_norm)

    else:
        ValueError(f"dist_type ({dist_type}) must be one of {list(DistType)}")


def get_axis_and_tiling(q: jnp.array) -> Tuple[Union[Tuple[int], int], Tuple[int]]:

    allowed_q_obs_dims = {1, 2, 3}

    if q.ndim == 3:
        num_obs, num_tokens, embedding_dim = q.shape
        tiling_rep_shape = (num_obs, 1, 1)
        norm_axis = (0, 1)
    elif q.ndim == 2:
        tiling_rep_shape = None
        norm_axis = (0, 1)
    elif q.ndim == 1:
        num_obs = len(q)
        tiling_rep_shape = (num_obs, 1)
        norm_axis = 0
    else:
        raise ValueError(f"q.ndim ({q.ndim}) must be one of: {allowed_q_obs_dims}")

    return norm_axis, tiling_rep_shape


def U_attractive(
    q,
    q_goal,
    gain: float,
    dist_type: DistType,
    norm_ord: int,
) -> float:

    return 0.5 * gain * distance(q, q_goal, dist_type=dist_type, norm_ord=norm_ord) ** 2


def U_repulsive(
    q,
    q_obstacles,
    dist_to_feel_repulsion: float,
    gain: float,
    obstacle_radius: float,
    dist_type: DistType,
    norm_ord: int,
) -> float:

    _, tiling_rep_shape = get_axis_and_tiling(q_obstacles)
    q_vec = jnp.tile(q, tiling_rep_shape)

    def repulsion_field(q_curr, q_obs) -> jnp.array:
        distances = distance(
            q_curr,
            q_obs,
            dist_type=dist_type,
            norm_ord=norm_ord,
        )

        # Interior of obstacles should be 0 distance
        # distances are measured from obstacle exterior
        distances = jnp.where(
            distances > obstacle_radius, distances - obstacle_radius, 0
        )
        repulsion_field = (1 / distances) - (1 / dist_to_feel_repulsion)

        # Clip repulsion field when you're greater than `dist_to_feel_repulsion` from obstacle
        # exterior
        repulsion_field = jnp.where(
            distances <= dist_to_feel_repulsion,
            repulsion_field,
            0,
        )

        return repulsion_field

    repulsion_vec = vmap(repulsion_field, in_axes=0, out_axes=0)

    return jnp.sum(0.5 * gain * repulsion_vec(q_vec, q_obstacles) ** 2, axis=0)


@partial(jit, static_argnames=U_static_argnames)
def U(
    q: jnp.array,
    q_goal: jnp.array,
    q_obstacles: jnp.array,
    dist_to_feel_repulsion: float,
    obstacle_radius: float,
    gain_attractive: float,
    gain_repulsive: float,
    dist_type: DistType,
    norm_ord: int,
) -> float:
    """computes scalar potential of object given location of goal and objects in object's space.

    Obstacles have a configurable uniform radius around them.
    If you traverse this potential your coordinates should stay outside of each obstacle's radius
    and reach the goal coordinates.

    Args:
        q: current object coordinates
        q_goal: coordinates of goal location. Shape = q.shape.
        q_obstacles: array of obstacle coordinates. Shape = (num_obstacles, *q.shape)
        dist_to_feel_repulsion: radial distance from each obstacle to. Depends on the dist_type and
            norm_ord.
        obstacle_radius: defines the radius around each obstacle. Depends on the dist_type and
            norm_ord.
            WARNING: Make sure you don't start with q inside an obstacle, or it won't ever move.
        gain_attractive: defines a strength of the attractive potential. Solutions aren't too
            sensitive to this, but will change with dist_type and norm_ord.
        gain_repulsive: defines the strength of the repulsive potential of each obstacle. Solutions
            aren't too sensitive to this, but will change with dist_type and norm_ord.
        dist_type: Type of distance metrics:
            "cosine": computes cosine distance (1 - cosine_similarity). If q are tensors, they're
                first flattened into vectors for angle computation.
                Distance ranges from:
                    min: 0
                    max: 2
            "norm": computes norm-based distance. Follow `numpy.linalg.norm()`'s guide for how to
                set `axis` and `ord` for tensor and vector `q`s.
                Distance ranges from:
                    min: 0
                    max: 2
        norm_ord: Order of the norm. inf means jax/numpy's inf object.
            ONLY USED IF dist_type = "norm".
            The default is None.
            follows numpy.linalg.norm():
            https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    Returns:
        the scalar potential of q's coordinates given the obstacles and the goal
    """

    if dist_to_feel_repulsion is None:
        dist_to_feel_repulsion = obstacle_radius

    return U_attractive(
        q=q,
        q_goal=q_goal,
        gain=gain_attractive,
        dist_type=dist_type,
        norm_ord=norm_ord,
    ) + U_repulsive(
        q=q,
        q_obstacles=q_obstacles,
        obstacle_radius=obstacle_radius,
        dist_to_feel_repulsion=dist_to_feel_repulsion,
        gain=gain_repulsive,
        dist_type=dist_type,
        norm_ord=norm_ord,
    )


grad_U = jit(grad(U), static_argnames=U_static_argnames)


@dataclass(eq=True, frozen=True)
class UConfig:
    dist_to_feel_repulsion: float = 0.1
    obstacle_radius: float = 0.05
    gain_attractive: float = 2e1
    gain_repulsive: float = 1e0
    dist_type: DistType = DistType.cosine
    # use jax/numpy default linalg.norm's ord parameter
    norm_ord: int = None


def compute_safe_path(
    q_start: jnp.array,
    q_goal: jnp.array,
    q_obstacles: jnp.array,
    grad_U: Callable = grad_U,
    grad_desc_rate: float = 1e-3,
    goal_radius: float = 1e-3,
    zero_grad_tol: float = 1e-5,
    random_jitter_mag_cov: float = 1e-5,
    N: int = 1000,
    U_config: UConfig = UConfig(),
    keep_full_history: bool = True,
    show_progbar: bool = True,
) -> List[jnp.array]:
    """computes a trajectory through encoding space from start to goal, avoiding obstacles

    Args:
        q_start: coordinates of start encoding "location". Shape = q_goal.shape.
        q_goal: coordinates of goal "location". Shape = q_start.shape.
        q_obstacles: array of obstacle coordinates. Shape = (num_obstacles, *q_start/goal.shape)
        grad_U: function computing gradient of the potential function, U for q values. Must have
            same args as salve.U.
        grad_desc_rate: fixed step size of each step of gradient descent.
        goal_radius: How far away, according to the chosen distance metric for grad_U (U_config),
            to consider a point on the trajectory "at" the goal.
            USED AS A STOPPING CONDITION.
        zero_grad_tol: gradient magnitude which is considered "no gradient".
            USED AS A STOPPING CONDITION.
        random_jitter_mag_cov: If > 0, applies a small amount of normal noise to the gradient at
            each update step. Critical for scenarios when the gradient is perfectly orthogonal
                to an obstacle that you want to get around.
        N: number of gradient descent steps before termination.
            USED AS A STOPPING CONDITION.
        U_config: configuration object for the potential, grad_U.
        keep_full_history: if False, only return the first and final coordinates of the trajectory.
        show_progbar: whether to show keras' Progbar.

    Returns:
        a list representing the encoding trajectory, containing the encoding at each gradient
            descent step.
    """

    key = random.PRNGKey(0)
    q = q_start
    q_history = [q]
    iters = list(range(0, N))

    if show_progbar:
        progbar = keras.utils.Progbar(N)

    for i in iters:

        delta_q = grad_U(
            q,
            q_goal,
            q_obstacles,
            U_config.dist_to_feel_repulsion,
            U_config.obstacle_radius,
            U_config.gain_attractive,
            U_config.gain_repulsive,
            U_config.dist_type,
            U_config.norm_ord,
        )

        # "Jiggles" your gradient a bit
        # this is very important for getting around an obstacle that you come to exactly
        # perpendicularly approach.
        cov = jnp.linalg.norm(delta_q) * random_jitter_mag_cov
        delta_q_jitter = random.normal(key, shape=delta_q.shape) * cov

        q -= grad_desc_rate * (delta_q + delta_q_jitter)

        # checking for stop conditions
        zero_gradient = jnp.linalg.norm(delta_q) < zero_grad_tol
        at_goal = (
            distance(
                q,
                q_goal,
                dist_type=U_config.dist_type,
                norm_ord=U_config.norm_ord,
            )
            <= goal_radius
        )

        if zero_gradient or at_goal:
            break

        if keep_full_history:
            q_history.append(q)

        if show_progbar:
            progbar.update(i)

    # still want to return the first and last q if we decide to not keep history
    if not keep_full_history:
        q_history.append(q)

    return q_history


def compute(
    f, q_collection, q_goal, q_obstacles, U_config: UConfig = UConfig()
) -> jnp.array:
    """does vectorized computation of f over coordinates q

    Args:
        f: the function to compute over q values
        q_collection: the list of coordinates at which to evaluate f
        q_goal: coordinates of goal location. Shape = q.shape.
        q_obstacles: array of obstacle coordinates. Shape = (num_obstacles, *q.shape)
        U_config: configuration object for the potential function

    Returns:
        an array of f values for each
    """

    U_config_kwargs = {
        "dist_to_feel_repulsion": U_config.dist_to_feel_repulsion,
        "obstacle_radius": U_config.obstacle_radius,
        "gain_attractive": U_config.gain_attractive,
        "gain_repulsive": U_config.gain_repulsive,
        "dist_type": U_config.dist_type,
        "norm_ord": U_config.norm_ord,
    }

    f_vmap_partial = vmap(partial(f, **U_config_kwargs), in_axes=(0, 0, 0), out_axes=0)
    n_evals, _, _ = q_collection.shape

    return f_vmap_partial(
        q_collection,
        jnp.tile(q_goal, (n_evals, 1, 1)),
        jnp.tile(q_obstacles, (n_evals, 1, 1, 1)),
    )


def plot_grid(
    images,
    grid_size=(2, 2),
    path=None,
    scale=2,
):

    import math

    fig = plt.figure(figsize=list(gs * scale for gs in grid_size))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.axis("off")
    images = images.astype(int)

    num_images = math.prod(grid_size)

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            index = row * num_images + col

            plt.subplot(*grid_size, index + 1)
            plt.imshow(images[index].astype("uint8"))
            plt.axis("off")
            plt.margins(x=0, y=0)

    if path is not None:
        plt.savefig(
            fname=path,
            pad_inches=0,
            bbox_inches="tight",
            transparent=False,
            dpi=60,
        )

    return fig


def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )


def main():

    U_config = UConfig(
        obstacle_radius=0.1,
        gain_attractive=10.0,
        gain_repulsive=1.0,
        dist_type=DistType.norm,
        norm_ord=2,
    )

    q_start = jnp.array([1.0, 1.0])
    q_goal = jnp.array([0.0, 0.0])
    q_obstacles = jnp.array([[[0.5, 0.5]], [[0.5, 0.1]], [[0.1, 0.5]]])

    q_history = compute_safe_path(
        q_start=q_start, q_goal=q_goal, q_obstacles=q_obstacles, U_config=U_config
    )

    print(q_history)

    grid_resolution = 50
    q0, q1 = jnp.mgrid[-1 : 1 : 1 / (grid_resolution), -1 : 1 : 1 / grid_resolution]
    q_vstacked = jnp.array(list([q] for q in zip(q0.ravel(), q1.ravel())))

    U_values = compute(U, q_vstacked, q_goal, q_obstacles)
    print(U_values)


if __name__ == "__main__":
    main()
