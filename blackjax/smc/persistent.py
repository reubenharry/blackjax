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
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

import blackjax.smc as smc
import blackjax.smc.from_mcmc as smc_from_mcmc
from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import update_and_take_last
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
import blackjax
import blackjax.smc as smc
import blackjax.smc.from_mcmc as smc_from_mcmc
import blackjax.smc.resampling as resampling
from blackjax import smc
from blackjax.base import SamplingAlgorithm
from blackjax.smc import extend_params
from blackjax.smc.base import SMCInfo, SMCState, update_and_take_last
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.smc.from_mcmc import unshared_parameters_and_step_fn


__all__ = ["PersistentSMCState", "init", "build_kernel", "as_top_level_api"]


class PersistentSMCState(NamedTuple):
    particles: ArrayTree
    likelihood: Array
    # Z: Array
    iteration: int
    Zs: Array


def init(particles: ArrayLikeTree):
    num_particles = particles.shape[0]
    particles = {0: particles}
    # num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    # weights = jnp.ones(num_particles) / num_particles
    return PersistentSMCState(particles, 0.0, iteration=0, Zs={0: 1})


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    # mcmc_init_fn: Callable,
    resampling_fn: Callable,
    # update_strategy: Callable = update_and_take_last,
    # update_particles_fn: Optional[Callable] = None,
    lmbda_schedule: Array,
) -> Callable:
    def kernel(
        rng_key: PRNGKey,
        state: PersistentSMCState,
        num_mcmc_steps: int,
        # mcmc_parameters: dict,
    ) -> tuple[PersistentSMCState, smc.base.SMCInfo]:
        kernel_key, resampling_key = jax.random.split(rng_key)

        flat_particles = jnp.stack(
            jax.tree.leaves(state.particles), axis=0
        )  # (num_iterations, num_particles, num_dimensions)
        flat_Zs = jnp.stack(jax.tree.leaves(state.Zs), axis=0)  # (num_iterations,)
        num_iterations, num_particles, num_dimensions = tuple(flat_particles.shape)

        ###################
        # calculate weights
        ###################

        ### todo: fix plogspace isue with Z
        numerator = lmbda_schedule[state.iteration + 1] * jax.vmap(
            jax.vmap(loglikelihood_fn)
        )(flat_particles)
        denom = jax.vmap(
            jax.vmap(
                lambda theta: -jnp.log(state.iteration)
                - jax.scipy.special.logsumexp(
                    jnp.array(
                        [
                            lmbda_schedule[i] * loglikelihood_fn(theta) - flat_Zs[i]
                            for i in range(state.iteration)
                        ]
                    )
                )
            )
        )(flat_particles)
        log_weights = numerator - denom  # (num_iterations, num_particles)

        new_Z = -jnp.log(
            (state.iteration) * num_particles
        ) + jax.scipy.special.logsumexp(log_weights)
        state = state._replace(Zs=state.Zs | {state.iteration + 1: new_Z})

        weights = log_weights - new_Z  # (num_iterations, num_particles)

        ###################
        # do resampling
        ###################

        ## consider using multinomial not systematic
        resampling_idx = resampling_fn(
            resampling_key,
            weights.reshape(num_iterations * num_particles),
            num_particles,
        )
        resampled_particles = flat_particles.reshape(
            num_iterations * num_particles, num_dimensions
        )[resampling_idx]

        ####################################
        # update particles with MCMC kernel
        ####################################

        def tempered_logposterior_fn(position: ArrayLikeTree) -> float:
            logprior = logprior_fn(position)
            tempered_loglikelihood = (
                loglikelihood_fn(position) * lmbda_schedule[state.iteration + 1]
            )
            return logprior + tempered_loglikelihood

        hmc_parameters = extend_params(
            dict(
                step_size=1e-2,
                inverse_mass_matrix=jnp.eye(1),
                num_integration_steps=100,
            )
        )
        unshared_mcmc_parameters, shared_mcmc_step_fn = unshared_parameters_and_step_fn(
            hmc_parameters, blackjax.hmc.build_kernel()
        )
        kernel, n_particles = update_and_take_last(
            blackjax.hmc.init,
            tempered_logposterior_fn,
            shared_mcmc_step_fn,
            num_mcmc_steps,
            num_particles,
        )
        print("n", n_particles, num_particles)

        new_particles, info = kernel(
            jax.random.split(kernel_key, num_particles),
            resampled_particles,
            unshared_mcmc_parameters,
        )
        return (
            state._replace(
                particles=state.particles | {state.iteration + 1: new_particles},
                iteration=state.iteration + 1,
            ),
            info,
        )

    return kernel

# def as_top_level_api(
#     logprior_fn: Callable,
#     loglikelihood_fn: Callable,
#     mcmc_step_fn: Callable,
#     mcmc_init_fn: Callable,
#     mcmc_parameters: dict,
#     resampling_fn: Callable,
#     num_mcmc_steps: Optional[int] = 10,
#     update_strategy=update_and_take_last,
#     update_particles_fn=None,
# ) -> SamplingAlgorithm:
#     """Implements the (basic) user interface for the Adaptive Persistent SMC kernel.

#     Parameters
#     ----------
#     logprior_fn
#         The log-prior function of the model we wish to draw samples from.
#     loglikelihood_fn
#         The log-likelihood function of the model we wish to draw samples from.
#     mcmc_step_fn
#         The MCMC step function used to update the particles.
#     mcmc_init_fn
#         The MCMC init function used to build a MCMC state from a particle position.
#     mcmc_parameters
#         The parameters of the MCMC step function.  Parameters with leading dimension
#         length of 1 are shared amongst the particles.
#     resampling_fn
#         The function used to resample the particles.
#     num_mcmc_steps
#         The number of times the MCMC kernel is applied to the particles per step.

#     Returns
#     -------
#     A ``SamplingAlgorithm``.

#     """

#     kernel = build_kernel(
#         logprior_fn,
#         loglikelihood_fn,
#         mcmc_step_fn,
#         mcmc_init_fn,
#         resampling_fn,
#         update_strategy,
#         update_particles_fn,
#     )

#     def init_fn(position: ArrayLikeTree, rng_key=None):
#         del rng_key
#         return init(position)

#     def step_fn(rng_key: PRNGKey, state, lmbda):
#         return kernel(
#             rng_key,
#             state,
#             num_mcmc_steps,
#             lmbda,
#             mcmc_parameters,
#         )

#     return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
