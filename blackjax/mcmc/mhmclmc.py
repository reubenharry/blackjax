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
"""Public API for the Metropolis Hastings Microcanonical Hamiltonian Monte Carlo (MHMCHMC) Kernel. This is closely related to the Microcanonical Langevin Monte Carlo (MCLMC) Kernel, which is an unadjusted method. This kernel adds a Metropolis-Hastings correction to the MCLMC kernel. It also only refreshes the momentum variable after each MH step, rather than during the integration of the trajectory. Hence "Hamiltonian" and not "Langevin"."""
from typing import Callable, Union

import jax
import jax.numpy as jnp

import blackjax.mcmc.integrators as integrators
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.hmc import HMCState, HMCInfo
from blackjax.mcmc.proposal import static_binomial_sampling
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_unit_vector

#__all__ = ["init", "build_kernel", "as_top_level_api"]


def build_kernel_malt(
    logdensity_fn,
    integrator: Callable = integrators.isokinetic_mclachlan,
    divergence_threshold: float = 1000,
    sqrt_diag_cov=1.0,
    L_proposal_factor = 1.25
):
    """Build a Dynamic MHMCHMC kernel where the number of integration steps is chosen randomly.

    Parameters
    ----------
    integrator
        The integrator to use to integrate the Hamiltonian dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """
    
    integrator_fn = integrators.with_isokinetic_maruyama(integrator(logdensity_fn=logdensity_fn, sqrt_diag_cov=sqrt_diag_cov))
    generate_proposal = adjusted_mclmc_proposal(integrator=integrator_fn, divergence_threshold=divergence_threshold)

    def kernel(
        rng_key: PRNGKey,
        state: HMCState,
        step_size: float,
        num_integration_steps: int = 10
    ) -> tuple[HMCState, HMCInfo]:
        """Generate a new sample with the MHMCHMC kernel."""


        key_momentum, key_integrator = jax.random.split(rng_key)
        momentum = generate_unit_vector(key_momentum, state.position)
        integrator_state = integrators.IntegratorState(state.position, momentum, state.logdensity, state.logdensity_grad)
        
        proposal, info, _ = generate_proposal(key_integrator, integrator_state, 
                                            step_size=step_size, 
                                            L_proposal= L_proposal_factor * num_integration_steps * step_size, 
                                            num_integration_steps=num_integration_steps)

        return HMCState(proposal.position, proposal.logdensity, proposal.logdensity_grad), info

    return kernel


def adjusted_mclmc_proposal(
    integrator: Callable,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = static_binomial_sampling,
) -> Callable:
    """Vanilla MHMCHMC algorithm.

    The algorithm integrates the trajectory applying a integrator
    `num_integration_steps` times in one direction to get a proposal and uses a
    Metropolis-Hastings acceptance step to either reject or accept this
    proposal. This is what people usually refer to when they talk about "the
    HMC algorithm".

    Parameters
    ----------
    integrator
        integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the integrator to build the trajectory
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the transition.

    """


    def build_trajectory(state, step_size, L_proposal, num_integration_steps, rng_key):
            
        def step(i, vars):
            state, kinetic_energy, rng_key = vars
            rng_key, next_rng_key = jax.random.split(rng_key)
            next_state, next_kinetic_energy = integrator(state, step_size, L_proposal, rng_key)

            return next_state, kinetic_energy + next_kinetic_energy, next_rng_key
        
        return jax.lax.fori_loop(0, num_integration_steps, step, (state, 0, rng_key))

    def generate(
        rng_key, state: integrators.IntegratorState,
        step_size, L_proposal, num_integration_steps,
    ) -> tuple[integrators.IntegratorState, HMCInfo, ArrayTree]:
        """Generate a new chain state."""
        end_state, kinetic_energy, rng_key = build_trajectory(
            state, step_size, L_proposal, num_integration_steps, rng_key
        )

        # note that this is the POTENTIAL energy only
        new_energy = -end_state.logdensity
        delta_energy = -state.logdensity + end_state.logdensity - kinetic_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_diverging = -delta_energy > divergence_threshold
        sampled_state, info = sample_proposal(rng_key, delta_energy, state, end_state)
        do_accept, p_accept, other_proposal_info = info

        info = HMCInfo(state.momentum, p_accept, do_accept, is_diverging, new_energy, end_state, num_integration_steps)

        return sampled_state, info, other_proposal_info

    return generate


def rescale(mu):
    """returns s, such that
     round(U(0, 1) * s + 0.5)
    has expected value mu.
    """
    k = jnp.floor(2 * mu - 1)
    x = k * (mu - 0.5 * (k + 1)) / (k + 1 - mu)
    return k + x
