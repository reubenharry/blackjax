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
"""Public API for the HMC Kernel"""
from typing import Callable, NamedTuple, Union

import jax

import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.trajectory import hmc_energy
from blackjax.mcmc.hmc import HMCState, init
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "uHMCInfo",
    "build_kernel",
    "as_top_level_api",
]


class uHMCInfo(NamedTuple):

    energy_change: ArrayTree



def static_integration(
    hmc_energy_fn,
    integrator: Callable,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating several times in one direction."""

    def integrate(initial_state: integrators.IntegratorState, step_size, num_integration_steps):
        directed_step_size = jax.tree_util.tree_map(
            lambda step_size: direction * step_size, step_size
        )

        def one_step(state, _):
            new_state= integrator(state, directed_step_size)
            energy_diff = hmc_energy_fn(new_state) - hmc_energy_fn(state)
            return new_state, energy_diff


        return jax.lax.scan(one_step, init= initial_state, length= num_integration_steps, xs= None)

    return integrate


def build_kernel(
    integrator: Callable = integrators.velocity_verlet
):

    def kernel(
        rng_key: PRNGKey,
        state: HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        num_integration_steps: int,
    ) -> tuple[HMCState, uHMCInfo]:
        """Generate a new sample with the HMC kernel."""

        metric = metrics.default_metric(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        proposal_generator = hmc_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            step_size,
            num_integration_steps
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = metric.sample_momentum(key_momentum, position)

        integrator_state = integrators.IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state)
        proposal = HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )

        return proposal, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: metrics.MetricTypes,
    num_integration_steps: int,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:

    kernel = build_kernel(integrator)

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            num_integration_steps,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def hmc_proposal(
    integrator: Callable,
    kinetic_energy: metrics.KineticEnergy,
    step_size: Union[float, ArrayLikeTree],
    num_integration_steps: int,
) -> Callable:
    
    hmc_energy_fn = hmc_energy(kinetic_energy)
    build_trajectory = static_integration(hmc_energy_fn, integrator)

    def generate(
        rng_key, state: integrators.IntegratorState
    ) -> tuple[integrators.IntegratorState, uHMCInfo]:
        """Generate a new chain state."""
        end_state, energy_diff = build_trajectory(state, step_size, num_integration_steps)        

        return end_state, uHMCInfo(energy_diff)

    return generate

