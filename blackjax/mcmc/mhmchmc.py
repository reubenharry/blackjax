# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for the MCLMC Kernel"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = ["RMCHMCState", "MCLMCInfo", "init", "build_kernel", "mclmc"]


from mclmc import Parameters, MCLMCInfo, full_refresh, update_position, update_momentum, minimal_norm


class RMCHMCState(NamedTuple):
    """State of the MCLMC algorithm."""
    
    t: float # time step (0., 1., 2., ....)
    x: Array # location in the sampling space
    l: float # - log p(x)
    g: Array # - grad log p(x)


def init(x_initial : ArrayLikeTree, logdensity_fn):

    grad_nlogp = jax.value_and_grad(lambda x : - logdensity_fn(x))
    l, g = grad_nlogp(x_initial)

    return RMCHMCState(0., x_initial, l, g)




def halton(t, max_bits=10):
    """for t= 0., 1., 2., ... it outputs halton sequence at that index (0.5, 0.25, 0.75, ...)
        taken from: https://github.com/tensorflow/probability/blob/main/discussion/snaper_hmc/SNAPER-HMC.ipynb"""
    float_index = jnp.asarray(t)
    bit_masks = 2**jnp.arange(max_bits, dtype=float_index.dtype)
    return jnp.einsum('i,i->', jnp.mod((float_index + 1) // bit_masks, 2), 0.5 / bit_masks)



def rescale(mu):
    """returns s, such that 
        round(U(0, 1) * s + 0.5)
       has expected value mu.    
    """
    k = jnp.floor(2 * mu -1)
    x = k * (mu - 0.5 *(k+1)) / (k + 1 - mu)
    return k + x
    

def trajectory_length(t, mu):
    s = rescale(mu)
    return jnp.rint(0.5 + halton(t) * s)



def proposal(hamiltonian_step, d):

    def prop(t, x, g, random_key, L, eps, sigma):
                
        #jiter the number of steps
        num_steps = jnp.rint(2 * halton(t) * L / eps).astype(int)
        
        #full momentum refreshment
        u = full_refresh(random_key, d)

        # do num_steps of the Hamiltonian dynamics

        def body(i, state):
            
            x, u, l, g, kinetic_energy = state
            xx, uu, ll, gg, kinetic_change = hamiltonian_step(x=x, u=u, g=g, eps=eps, sigma = sigma)

            return xx, uu, ll, gg, kinetic_energy + kinetic_change
        
        xx, uu, ll, gg, kinetic_change = jax.fori_loop(0, num_steps, body, (x, u, 0., g, 0.))
        
        return xx, ll, gg, kinetic_change

    return prop


def build_kernel(grad_nlogp, d, integrator, transform, params):

    L, eps, sigma = params

    hamiltonian_step, _ = integrator(T= update_position(grad_nlogp), V= update_momentum(d), d= d)
    get_proposal = proposal(hamiltonian_step, d)
    
    def kernel(rng_key : PRNGKey, state : RMCHMCState) -> tuple[RMCHMCState, MCLMCInfo]:
        
        key1, key2 = jax.random.split(rng_key)
            
        t, x, l, g = state
        xx, ll, gg, kinetic_change = get_proposal(t, x, g, key1, L, eps, sigma)
        de = kinetic_change + ll - l
        
        # accept/reject

        acc_prob = jnp.clip(jnp.exp(-de), 0, 1)
        accept = jax.random.bernoulli(key2, acc_prob)
        xx, ll, gg = jax.tree_util.tree_map(lambda new, old: jax.lax.select(accept, new, old), (xx, ll, gg), (x, l, g))
               
        return RMCHMCState(t + 1., xx, ll, gg), MCLMCInfo(transform(xx), ll, de)

    return kernel


class rmchmc:
    """todo: add documentation"""

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        d : int,
        transform : Callable,
        params : Parameters,
        *,
        integrator = minimal_norm,
    ) -> SamplingAlgorithm:
                
        grad_nlogp = jax.value_and_grad(lambda x : - logdensity_fn(x))

        kernel = cls.build_kernel(grad_nlogp, d, integrator, transform, params)

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
            )

        return SamplingAlgorithm(init_fn, step_fn)
    

# import blackjax.mcmc.integrators as integrators
# import blackjax.mcmc.metrics as metrics
# import blackjax.mcmc.trajectory as trajectory
# from blackjax.base import SamplingAlgorithm
# from blackjax.mcmc.proposal import safe_energy_diff, static_binomial_sampling
# from blackjax.mcmc.trajectory import hmc_energy
# from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

# __all__ = [
#     "init",
#     "init_dynamic",
#     "build_kernel",
#     "build_dynamic_kernel",
#     "mhmchmc",
#     # "dynamic_hmc",
# ]




# def init(position: ArrayLikeTree, logdensity_fn: Callable):
#     logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
#     return HMCState(position, logdensity, logdensity_grad)


# def init_dynamic(
#     position: ArrayLikeTree, logdensity_fn: Callable, random_generator_arg: Array
# ):
#     logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
#     return DynamicHMCState(position, logdensity, logdensity_grad, random_generator_arg)


# def build_kernel(
#     integrator: Callable = integrators.velocity_verlet,
#     divergence_threshold: float = 1000,
# ):
#     """Build a kernel for MH-MCHMC.

#     Parameters
#     ----------
#     integrator
#         The symplectic integrator to use to integrate the Hamiltonian dynamics.
#     divergence_threshold
#         Value of the difference in energy above which we consider that the transition is divergent.

#     Returns
#     -------
#     A kernel that takes a rng_key and a Pytree that contains the current state
#     of the chain and that returns a new state of the chain along with
#     information about the transition.

#     """

#     def kernel(
#         rng_key: PRNGKey,
#         state: HMCState,
#         logdensity_fn: Callable,
#         step_size: float,
#         inverse_mass_matrix: Array,
#         num_integration_steps: int,
#     ) -> tuple[HMCState, HMCInfo]:
#         """Generate a new sample with the HMC kernel."""

#         # momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
#         #     inverse_mass_matrix
#         # )
#         symplectic_integrator = integrator(logdensity_fn, kinetic_energy_fn)
#         proposal_generator = hmc_proposal(
#             symplectic_integrator,
#             kinetic_energy_fn,
#             step_size,
#             num_integration_steps,
#             divergence_threshold,
#         )

#         key_momentum, key_integrator = jax.random.split(rng_key, 2)

#         position, logdensity, logdensity_grad = state
#         momentum = momentum_generator(key_momentum, position)

#         integrator_state = integrators.IntegratorState(
#             position, momentum, logdensity, logdensity_grad
#         )
#         proposal, info, _ = proposal_generator(key_integrator, integrator_state)
#         proposal = HMCState(
#             proposal.position, proposal.logdensity, proposal.logdensity_grad
#         )

#         return proposal, info

#     return kernel


# def build_dynamic_kernel(
#     integrator: Callable = integrators.velocity_verlet,
#     divergence_threshold: float = 1000,
#     next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
#     integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
# ):
#     """Build a Dynamic HMC kernel where the number of integration steps is chosen randomly.

#     Parameters
#     ----------
#     integrator
#         The symplectic integrator to use to integrate the Hamiltonian dynamics.
#     divergence_threshold
#         Value of the difference in energy above which we consider that the transition is divergent.
#     next_random_arg_fn
#         Function that generates the next `random_generator_arg` from its previous value.
#     integration_steps_fn
#         Function that generates the next pseudo or quasi-random number of integration steps in the
#         sequence, given the current `random_generator_arg`. Needs to return an `int`.

#     Returns
#     -------
#     A kernel that takes a rng_key and a Pytree that contains the current state
#     of the chain and that returns a new state of the chain along with
#     information about the transition.

#     """
#     hmc_base = build_kernel(integrator, divergence_threshold)

#     def kernel(
#         rng_key: PRNGKey,
#         state: DynamicHMCState,
#         logdensity_fn: Callable,
#         step_size: float,
#         inverse_mass_matrix: Array,
#         **integration_steps_kwargs,
#     ) -> tuple[DynamicHMCState, HMCInfo]:
#         """Generate a new sample with the HMC kernel."""
#         num_integration_steps = integration_steps_fn(
#             state.random_generator_arg, **integration_steps_kwargs
#         )
#         hmc_state = HMCState(state.position, state.logdensity, state.logdensity_grad)
#         hmc_proposal, info = hmc_base(
#             rng_key,
#             hmc_state,
#             logdensity_fn,
#             step_size,
#             inverse_mass_matrix,
#             num_integration_steps,
#         )
#         next_random_arg = next_random_arg_fn(state.random_generator_arg)
#         return (
#             DynamicHMCState(
#                 hmc_proposal.position,
#                 hmc_proposal.logdensity,
#                 hmc_proposal.logdensity_grad,
#                 next_random_arg,
#             ),
#             info,
#         )

#     return kernel


# class hmc:
#     """Implements the (basic) user interface for the HMC kernel.

#     The general hmc kernel builder (:meth:`blackjax.mcmc.hmc.build_kernel`, alias `blackjax.hmc.build_kernel`) can be
#     cumbersome to manipulate. Since most users only need to specify the kernel
#     parameters at initialization time, we provide a helper function that
#     specializes the general kernel.

#     We also add the general kernel and state generator as an attribute to this class so
#     users only need to pass `blackjax.hmc` to SMC, adaptation, etc. algorithms.

#     Examples
#     --------

#     A new HMC kernel can be initialized and used with the following code:

#     .. code::

#         hmc = blackjax.hmc(logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
#         state = hmc.init(position)
#         new_state, info = hmc.step(rng_key, state)

#     Kernels are not jit-compiled by default so you will need to do it manually:

#     .. code::

#        step = jax.jit(hmc.step)
#        new_state, info = step(rng_key, state)

#     Should you need to you can always use the base kernel directly:

#     .. code::

#        import blackjax.mcmc.integrators as integrators

#        kernel = blackjax.hmc.build_kernel(integrators.mclachlan)
#        state = blackjax.hmc.init(position, logdensity_fn)
#        state, info = kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)

#     Parameters
#     ----------
#     logdensity_fn
#         The log-density function we wish to draw samples from.
#     step_size
#         The value to use for the step size in the symplectic integrator.
#     inverse_mass_matrix
#         The value to use for the inverse mass matrix when drawing a value for
#         the momentum and computing the kinetic energy.
#     num_integration_steps
#         The number of steps we take with the symplectic integrator at each
#         sample step before returning a sample.
#     divergence_threshold
#         The absolute value of the difference in energy between two states above
#         which we say that the transition is divergent. The default value is
#         commonly found in other libraries, and yet is arbitrary.
#     integrator
#         (algorithm parameter) The symplectic integrator to use to integrate the trajectory.\

#     Returns
#     -------
#     A ``SamplingAlgorithm``.
#     """

#     init = staticmethod(init)
#     build_kernel = staticmethod(build_kernel)

#     def __new__(  # type: ignore[misc]
#         cls,
#         logdensity_fn: Callable,
#         step_size: float,
#         inverse_mass_matrix: Array,
#         num_integration_steps: int,
#         *,
#         divergence_threshold: int = 1000,
#         integrator: Callable = integrators.velocity_verlet,
#     ) -> SamplingAlgorithm:
#         kernel = cls.build_kernel(integrator, divergence_threshold)

#         def init_fn(position: ArrayLikeTree):
#             return cls.init(position, logdensity_fn)

#         def step_fn(rng_key: PRNGKey, state):
#             return kernel(
#                 rng_key,
#                 state,
#                 logdensity_fn,
#                 step_size,
#                 inverse_mass_matrix,
#                 num_integration_steps,
#             )

#         return SamplingAlgorithm(init_fn, step_fn)


# class dynamic_hmc:
#     """Implements the (basic) user interface for the dynamic HMC kernel.

#     Parameters
#     ----------
#     logdensity_fn
#         The log-density function we wish to draw samples from.
#     step_size
#         The value to use for the step size in the symplectic integrator.
#     inverse_mass_matrix
#         The value to use for the inverse mass matrix when drawing a value for
#         the momentum and computing the kinetic energy.
#     divergence_threshold
#         The absolute value of the difference in energy between two states above
#         which we say that the transition is divergent. The default value is
#         commonly found in other libraries, and yet is arbitrary.
#     integrator
#         (algorithm parameter) The symplectic integrator to use to integrate the trajectory.
#     next_random_arg_fn
#         Function that generates the next `random_generator_arg` from its previous value.
#     integration_steps_fn
#         Function that generates the next pseudo or quasi-random number of integration steps in the
#         sequence, given the current `random_generator_arg`.


#     Returns
#     -------
#     A ``SamplingAlgorithm``.
#     """

#     init = staticmethod(init_dynamic)
#     build_kernel = staticmethod(build_dynamic_kernel)

#     def __new__(  # type: ignore[misc]
#         cls,
#         logdensity_fn: Callable,
#         step_size: float,
#         inverse_mass_matrix: Array,
#         *,
#         divergence_threshold: int = 1000,
#         integrator: Callable = integrators.velocity_verlet,
#         next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1],
#         integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10),
#     ) -> SamplingAlgorithm:
#         kernel = cls.build_kernel(
#             integrator, divergence_threshold, next_random_arg_fn, integration_steps_fn
#         )

#         def init_fn(position: ArrayLikeTree, random_generator_arg: Array):
#             return cls.init(position, logdensity_fn, random_generator_arg)

#         def step_fn(rng_key: PRNGKey, state):
#             return kernel(
#                 rng_key,
#                 state,
#                 logdensity_fn,
#                 step_size,
#                 inverse_mass_matrix,
#             )

#         return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]


# def hmc_proposal(
#     integrator: Callable,
#     kinetic_energy: Callable,
#     step_size: Union[float, ArrayLikeTree],
#     num_integration_steps: int = 1,
#     divergence_threshold: float = 1000,
#     *,
#     sample_proposal: Callable = static_binomial_sampling,
# ) -> Callable:
#     """Vanilla HMC algorithm.

#     The algorithm integrates the trajectory applying a symplectic integrator
#     `num_integration_steps` times in one direction to get a proposal and uses a
#     Metropolis-Hastings acceptance step to either reject or accept this
#     proposal. This is what people usually refer to when they talk about "the
#     HMC algorithm".

#     Parameters
#     ----------
#     integrator
#         Symplectic integrator used to build the trajectory step by step.
#     kinetic_energy
#         Function that computes the kinetic energy.
#     step_size
#         Size of the integration step.
#     num_integration_steps
#         Number of times we run the symplectic integrator to build the trajectory
#     divergence_threshold
#         Threshold above which we say that there is a divergence.

#     Returns
#     -------
#     A kernel that generates a new chain state and information about the transition.

#     """
#     build_trajectory = trajectory.static_integration(integrator)
#     hmc_energy_fn = hmc_energy(kinetic_energy)

#     def generate(
#         rng_key, state: integrators.IntegratorState
#     ) -> tuple[integrators.IntegratorState, HMCInfo, ArrayTree]:
#         """Generate a new chain state."""
#         end_state = build_trajectory(state, step_size, num_integration_steps)
#         end_state = flip_momentum(end_state)
#         proposal_energy = hmc_energy_fn(state)
#         new_energy = hmc_energy_fn(end_state)
#         delta_energy = safe_energy_diff(proposal_energy, new_energy)
#         is_diverging = -delta_energy > divergence_threshold
#         sampled_state, info = sample_proposal(rng_key, delta_energy, state, end_state)
#         do_accept, p_accept, other_proposal_info = info

#         info = HMCInfo(
#             state.momentum,
#             p_accept,
#             do_accept,
#             is_diverging,
#             new_energy,
#             end_state,
#             num_integration_steps,
#         )

#         return sampled_state, info, other_proposal_info

#     return generate


# def flip_momentum(
#     state: integrators.IntegratorState,
# ) -> integrators.IntegratorState:
#     """Flip the momentum at the end of the trajectory.

#     To guarantee time-reversibility (hence detailed balance) we
#     need to flip the last state's momentum. If we run the hamiltonian
#     dynamics starting from the last state with flipped momentum we
#     should indeed retrieve the initial state (with flipped momentum).

#     """
#     flipped_momentum = jax.tree_util.tree_map(lambda m: -1.0 * m, state.momentum)
#     return integrators.IntegratorState(
#         state.position,
#         flipped_momentum,
#         state.logdensity,
#         state.logdensity_grad,
#     )
