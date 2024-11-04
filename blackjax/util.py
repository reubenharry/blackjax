"""Utility functions for BlackJax."""
from functools import partial
from typing import Callable, Union

import jax.numpy as jnp
from jax import jit, lax, vmap, device_put
from jax.flatten_util import ravel_pytree
from jax.random import normal, split
from jax.tree_util import tree_leaves, tree_map

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map

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


def run_inference_algorithm(
    rng_key: PRNGKey,
    initial_state_or_position: ArrayLikeTree,
    inference_algorithm: Union[SamplingAlgorithm, VIAlgorithm],
    num_steps: int,
    progress_bar: bool = False,
    transform: Callable = lambda x: x,
) -> tuple[State, State, Info]:
    """Wrapper to run an inference algorithm.

    Note that this utility function does not work for Stochastic Gradient MCMC samplers
    like sghmc, as SG-MCMC samplers require additional control flow for batches of data
    to be passed in during each sample.

    Parameters
    ----------
    rng_key
        The random state used by JAX's random numbers generator.
    initial_state_or_position
        The initial state OR the initial position of the inference algorithm. If an initial position
        is passed in, the function will automatically convert it into an initial state.
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

    Returns
    -------
    Tuple[State, State, Info]
        1. The final state of the inference algorithm.
        2. The trace of states of the inference algorithm (contains the MCMC samples).
        3. The trace of the info of the inference algorithm for diagnostics.
    """
    init_key, sample_key = split(rng_key, 2)
    # try:
    #     initial_state = inference_algorithm.init(initial_state_or_position, init_key)
    #except (TypeError, ValueError, AttributeError, IndexError):
    # We assume initial_state is already in the right format.
    initial_state = initial_state_or_position

    keys = split(sample_key, num_steps)

    @jit
    def _one_step(state, xs):
        _, rng_key = xs
        state, info = inference_algorithm.step(rng_key, state)
        return state, (transform(state), info)

    if progress_bar:
        one_step = progress_bar_scan(num_steps)(_one_step)
    else:
        one_step = _one_step

    xs = (jnp.arange(num_steps), keys)
    final_state, (state_history, info_history) = lax.scan(one_step, initial_state, xs)
    return final_state, state_history, info_history


def eca_step(kernel, summary_statistics_fn, adaptation_update, num_chains):

    def step(state_all, xs):
        """This function operates on a single device."""
        state, adaptation_state = state_all # state is an array of states, one for each chain on this device. adaptation_state is the same for all chains, so it is not an array.
        _, keys_sampling, key_adaptation = xs # keys_sampling.shape = (chains_per_device, )
        
        # update the state of all chains on this device
        state, info = vmap(kernel, (0, 0, None))(keys_sampling, state, adaptation_state)
        
        # combine all the chains to compute expectation values
        theta = vmap(summary_statistics_fn, (0, 0))(state, info)
        Etheta = tree_map(lambda theta: lax.psum(jnp.sum(theta, axis= 0), axis_name= 'chains') / num_chains, theta)

        # use these to adapt the hyperparameters of the dynamics
        adaptation_state, info_to_be_stored = adaptation_update(adaptation_state, Etheta, key_adaptation)
        
        return (state, adaptation_state), info_to_be_stored

    return step


def run_eca(rng_key, initial_state, kernel, adaptation, num_steps, num_chains, mesh):
 
    step = eca_step(kernel, adaptation.summary_statistics_fn, adaptation.update, num_chains)


    def all_steps(initial_state, keys_sampling, keys_adaptation):
        """This function operates on a single device. key is a random key for this device."""
        
        initial_state_all = (initial_state, adaptation.initial_state)
        
        # run sampling
        xs = (jnp.arange(num_steps), keys_sampling.T, keys_adaptation) # keys for all steps that will be performed. keys_sampling.shape = (num_steps, chains_per_device), keys_adaptation.shape = (num_steps, )
        
        final_state_all, info_history = lax.scan(step, initial_state_all, xs)
        final_state, final_adaptation_state = final_state_all
        return final_state, final_adaptation_state, info_history # info history is composed of averages over all chains, so it is a couple of scalars


    p, pscalar = PartitionSpec('chains'), PartitionSpec()
    parallel_execute = shard_map(all_steps, 
                            mesh= mesh,
                            in_specs= (p, p, pscalar), 
                            out_specs= (p, pscalar, pscalar),
                            check_rep=False
                            )
    
    # produce all random keys that will be needed
    key_sampling, key_adaptation = split(rng_key)
    keys_adaptation = split(key_adaptation, num_steps)
    distribute_keys = lambda key, shape: device_put(split(key, shape), NamedSharding(mesh, p)) # random keys, distributed across devices
    keys_sampling = distribute_keys(key_sampling, (num_chains, num_steps))

    
    # run sampling in parallel
    final_state, final_adaptation_state, info_history = parallel_execute(initial_state, keys_sampling, keys_adaptation)
    
    return final_state, final_adaptation_state, info_history




def ensemble_execute_fn(func, rng_key, num_chains, mesh, 
                        x= None, 
                        args= None,
                        summary_statistics_fn= lambda y: 0.,
    ):
    """Given a sequential function 
        func(rng_key, x, args) = y,
       evaluate it with an ensemble and also compute some summary statistics E[theta(y)], where expectation is taken over ensemble.
       Args:
            x: array distributed over all decvices
            args: additional arguments for func, not distributed.
            summary_statistics_fn: operates on a single member of ensemble and returns some summary statistics.
            rng_key: a single random key, which will then be split, such that each member of an ensemble will get a different random key. 
        
       Returns:
            y: array distributed over all decvices. Need not be of the same shape as x.
            Etheta: expected values of the summary statistics
    """
    p, pscalar = PartitionSpec('chains'), PartitionSpec()
    
    if x == None:
        X= device_put(jnp.zeros(num_chains), NamedSharding(mesh, p)) # random keys, distributed across devices

    else: 
        X= x
    
    adaptation_update= lambda useless_param1, Etheta, useless_param2: (Etheta, None)
    
    _F = eca_step(func, lambda y, _: summary_statistics_fn(y), adaptation_update, num_chains)

    def F(x, keys):
        """This function operates on a single device. key is a random key for this device."""
        y, summary_statistics = _F((x, args), (None, keys, None))[0]
        return y, summary_statistics

    parallel_execute = shard_map(F, 
                            mesh= mesh,
                            in_specs= (p, p), 
                            out_specs= (p, pscalar),
                            check_rep=False
                            )
    
    keys = device_put(split(rng_key, num_chains), NamedSharding(mesh, p)) # random keys, distributed across devices
    # apply F in parallel
    return parallel_execute(X, keys)
    
    