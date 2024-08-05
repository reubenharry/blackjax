"""Utility functions for BlackJax."""

from functools import partial
from typing import Callable, Union

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.flatten_util import ravel_pytree
from jax.random import normal, split
from jax.tree_util import tree_leaves

from blackjax.base import Info, SamplingAlgorithm, State, VIAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey


@partial(jit, static_argnames=("precision",), inline=True)
def linear_map(diag_or_dense_a, b, *, precision="highest"):
    """Perform a linear map of the form y = Ax.

    Dispatch matrix multiplication to either jnp.dot or jnp.multiply.

    Unlike jax.numpy.dot, this function output an Array that match the dtype
    and shape of the 2nd input:
    - diag_or_dense_a is a scalar or 1d vector, `diag_or_dense_a * b` is returned
    - diag_or_dense_a is a 2d matrix, `diag_or_dense_a @ b` is returned

    Note that unlike jax.numpy.dot, here we defaults to full (highest)
    precision. This is more useful for numerical algorithms and will be the
    default for jax.numpy in the future:
    https://github.com/google/jax/pull/7859

    Parameters
    ----------
    diag_or_dense_a:
        A diagonal (1d vector) or dense matrix (2d square matrix).
    b:
        A vector.
    precision:
        The precision of the computation. See jax.lax.dot_general for
        more details.

    Returns
    -------
        The result vector of the matrix multiplication.
    """
    dtype = jnp.result_type(diag_or_dense_a.dtype, b.dtype)
    diag_or_dense_a = diag_or_dense_a.astype(dtype)
    b = b.astype(dtype)
    ndim = jnp.ndim(diag_or_dense_a)

    if ndim <= 1:
        return lax.mul(diag_or_dense_a, b)
    else:
        return lax.dot(diag_or_dense_a, b, precision=precision)


# TODO(https://github.com/blackjax-devs/blackjax/issues/376)
# Refactor this function to not use ravel_pytree might be more performant.
def generate_gaussian_noise(
    rng_key: PRNGKey,
    position: ArrayLikeTree,
    mu: Union[float, Array] = 0.0,
    sigma: Union[float, Array] = 1.0,
) -> ArrayTree:
    """Generate N(mu, sigma) noise with output structure that match a given PyTree.

    Parameters
    ----------
    rng_key:
        The pseudo-random number generator key used to generate random numbers.
    position:
        PyTree that the structure the output should to match.
    mu:
        The mean of the Gaussian distribution.
    sigma:
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    Gaussian noise following N(mu, sigma) that match the structure of position.
    """
    p, unravel_fn = ravel_pytree(position)
    sample = normal(rng_key, shape=p.shape, dtype=p.dtype)
    return unravel_fn(mu + linear_map(sigma, sample))


def generate_unit_vector(
    rng_key: PRNGKey,
    position: ArrayLikeTree,
) -> Array:
    """Generate a random unit vector with output structure that match a given PyTree.

    Parameters
    ----------
    rng_key:
        The pseudo-random number generator key used to generate random numbers.
    position:
        PyTree that the structure the output should to match.

    Returns
    -------
    Random unit vector that match the structure of position.
    """
    p, unravel_fn = ravel_pytree(position)
    sample = normal(rng_key, shape=p.shape, dtype=p.dtype)
    return unravel_fn(sample / jnp.linalg.norm(sample))


def pytree_size(pytree: ArrayLikeTree) -> int:
    """Return the dimension of the flatten PyTree."""
    return sum(jnp.size(value) for value in tree_leaves(pytree))


def index_pytree(input_pytree: ArrayLikeTree) -> ArrayTree:
    """Builds a PyTree with elements indicating its corresponding index on a flat array.

    Various algorithms in BlackJAX take as input a 1 or 2 dimensional array which somehow
    affects the sampling or approximation of a PyTree. For instance, in HMC a 1 or 2
    dimensional inverse mass matrix is used when simulating Hamilonian dynamics on
    PyTree position and momentum variables. It is usually unclear how the elements of the
    array interact with the PyTree. This function demonstrates how all algorithms map an
    array to a PyTree of equivalent dimension.

    The function returns the index of a 1 dimensional array corresponding to each element of
    the PyTree. This way the user can tell which element in the PyTree corresponds to which
    column (and row) of a 1 dimensional (or 2 dimensional) array.

    Parameters
    ----------
    input_pytree:
        Example PyTree.

    Returns
    -------
    PyTree mapping each individual element of an arange array to elements in the PyTree.
    """
    flat_input, unravel_fn = ravel_pytree(input_pytree)
    (dim_input,) = flat_input.shape
    array = jnp.arange(dim_input, dtype=flat_input.dtype)
    return unravel_fn(array)


# def run_inference_algorithm(
#     rng_key: PRNGKey,
#     inference_algorithm: Union[SamplingAlgorithm, VIAlgorithm],
#     num_steps: int,
#     initial_state: ArrayLikeTree = None,
#     initial_position: ArrayLikeTree = None,
#     progress_bar: bool = False,
#     transform: Callable = lambda x: x,
# ) -> tuple[State, State, Info]:
    
    
#     """Wrapper to run an inference algorithm.

#     Note that this utility function does not work for Stochastic Gradient MCMC samplers
#     like sghmc, as SG-MCMC samplers require additional control flow for batches of data
#     to be passed in during each sample.

#     Parameters
#     ----------
#     rng_key
#         The random state used by JAX's random numbers generator.
#     initial_state_or_position
#         The initial state OR the initial position of the inference algorithm. If an initial position
#         is passed in, the function will automatically convert it into an initial state.
#     inference_algorithm
#         One of blackjax's sampling algorithms or variational inference algorithms.
#     num_steps
#         Number of MCMC steps.
#     progress_bar
#         Whether to display a progress bar.
#     transform
#         A transformation of the trace of states to be returned. This is useful for
#         computing determinstic variables, or returning a subset of the states.
#         By default, the states are returned as is.

#     Returns
#     -------
#     Tuple[State, State, Info]
#         1. The final state of the inference algorithm.
#         2. The trace of states of the inference algorithm (contains the MCMC samples).
#         3. The trace of the info of the inference algorithm for diagnostics.
#     """

#     if initial_state is None and initial_position is None:
#         raise ValueError("Either initial_state or initial_position must be provided.")
#     if initial_state is not None and initial_position is not None:
#         raise ValueError(
#             "Only one of initial_state or initial_position must be provided."
#         )
    
#     init_key, sample_key = split(rng_key, 2)
#     if initial_position is not None:
#         initial_state = inference_algorithm.init(initial_position, init_key)
    
#     keys = split(sample_key, num_steps)

#     @jit
#     def _one_step(state, xs):
#         _, rng_key = xs
#         state, info = inference_algorithm.step(rng_key, state)
#         return state, transform(state, info)

#     if progress_bar:
#         one_step = progress_bar_scan(num_steps)(_one_step)
#     else:
#         one_step = _one_step

#     xs = (jnp.arange(num_steps), keys)
#     final_state, (state_history, info_history) = lax.scan(one_step, initial_state, xs)
#     return final_state, state_history, info_history

def run_inference_algorithm(
    rng_key: PRNGKey,
    inference_algorithm: Union[SamplingAlgorithm, VIAlgorithm],
    num_steps: int,
    initial_state: ArrayLikeTree = None,
    initial_position: ArrayLikeTree = None,
    progress_bar: bool = False,
    transform: Callable = lambda state, info: (state, info),
) -> tuple:
    """Wrapper to run an inference algorithm.

    Note that this utility function does not work for Stochastic Gradient MCMC samplers
    like sghmc, as SG-MCMC samplers require additional control flow for batches of data
    to be passed in during each sample.

    Parameters
    ----------
    rng_key
        The random state used by JAX's random numbers generator.
    initial_state
        The initial state of the inference algorithm.
    initial_position
        The initial position of the inference algorithm. This is used when the initial state is not provided.
    inference_algorithm
        One of blackjax's sampling algorithms or variational inference algorithms.
    num_steps
        Number of MCMC steps.
    progress_bar
        Whether to display a progress bar.
    transform
        A transformation of the trace of states to be returned. This is useful for
        computing determinstic variables, or returning a subset of the states.
        By default, the states are returned as is.
    return_state_history
        if False, `run_inference_algorithm` will only return an expectation of the value of transform, and return that average instead of the full set of samples. This is useful when memory is a bottleneck.

    Returns
    -------
        1. The final state.
        2. The trace of the transform(state)
        3. The trace of the info of the inference algorithm for diagnostics.
    """

    if initial_state is None and initial_position is None:
        raise ValueError("Either initial_state or initial_position must be provided.")
    if initial_state is not None and initial_position is not None:
        raise ValueError(
            "Only one of initial_state or initial_position must be provided."
        )

    rng_key, init_key = split(rng_key, 2)
    if initial_position is not None:
        initial_state = inference_algorithm.init(initial_position, init_key)

    keys = split(rng_key, num_steps)

    def one_step(state, xs):
        _, rng_key = xs
        state, info = inference_algorithm.step(rng_key, state)
        return state, transform(state, info)
        
    if progress_bar:
        one_step = progress_bar_scan(num_steps)(one_step)

    xs = jnp.arange(num_steps), keys
    final_state, history = lax.scan(one_step, initial_state, xs)
    return final_state, history


def store_only_expectation_values(sampling_algorithm, state_transform= lambda x: x, exp_vals_transform= lambda x: x):
    """Takes a sampling algorithm and constructs from it a new sampling algorithm object. The new sampling algorithm has the same 
        kernel but only stores the streaming expectation values of some observables, not the full states; to save memory.

       It saves exp_vals_transform(E[state_transform(x)]) at each step i, where expectation is computed with samples up to i-th sample.
       
       Example:

       .. code::

            init_key, state_key, run_key = jax.random.split(jax.random.PRNGKey(0),3)
            model = StandardNormal(2)
            initial_position = model.sample_init(init_key)
            initial_state = blackjax.mcmc.mclmc.init(
                position=initial_position, logdensity_fn=model.logdensity_fn, rng_key=state_key
            )
            integrator_type = "mclachlan"
            L = 1.0
            step_size = 0.1
            num_steps = 4

            integrator = map_integrator_type_to_integrator['mclmc'][integrator_type]
            state_transform = lambda state: x.position
            memory_efficient_sampling_alg, transform = store_only_expectation_values(
                sampling_algorithm=sampling_alg,
                state_transform=state_transform)
            
            initial_state = memory_efficient_sampling_alg.init(initial_state)
                
            final_state, trace_at_every_step = run_inference_algorithm(

                rng_key=run_key,
                initial_state=initial_state,
                inference_algorithm=memory_efficient_sampling_alg,
                num_steps=num_steps,
                transform=transform,
                progress_bar=True,
            )           
    """
    
    def init_fn(state):
        averaging_state = (0., state_transform(state))
        return (state, averaging_state)

    def update_fn(rng_key, state_full):
        state, averaging_state = state_full
        state, info = sampling_algorithm.step(rng_key, state) # update the state with the sampling algorithm
        averaging_state = streaming_average_update(state_transform(state), averaging_state) # update the expectation value with the Kalman filter
        return (state, averaging_state), info
    
    def transform(full_state, info):
        exp_vals = full_state[1][1]
        return exp_vals_transform(exp_vals), info

    return SamplingAlgorithm(init_fn, update_fn), transform
    


def streaming_average_update(expectation, streaming_avg, weight=1.0, zero_prevention=0.0):
    """Compute the streaming average of a function O(x) using a weight.
    Parameters:
    ----------
        expectation
            the value of the expectation at the current timestep
        streaming_avg
            tuple of (total, average) where total is the sum of weights and average is the current average
        weight
            weight of the current state
        zero_prevention
            small value to prevent division by zero
    Returns:
    ----------
        new streaming average
    """

    flat_expectation, unravel_fn = ravel_pytree(expectation)
    total, average = streaming_avg
    flat_average, _ = ravel_pytree(average)
    average = (total * flat_average + weight * flat_expectation) / (
        total + weight + zero_prevention
    )
    total += weight
    streaming_avg = (total, unravel_fn(average))
    return streaming_avg
