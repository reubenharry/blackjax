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
"""Public API for the Underdamped Langevin Kernel"""
from typing import Callable, NamedTuple

from blackjax.mcmc import metrics
import jax

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import (
    IntegratorState,
    velocity_verlet,
    with_maruyama,
)
from blackjax.types import ArrayLike, PRNGKey
from blackjax.util import generate_unit_vector, pytree_size

__all__ = ["UnderdampedLangevinInfo", "init", "build_kernel", "as_top_level_api"]


class UnderdampedLangevinInfo(NamedTuple):
    """
    Additional information on the UnderdampedLangevin transition.

    logdensity
        The log-density of the distribution at the current step of the UnderdampedLangevin chain.
    kinetic_change
        The difference in kinetic energy between the current and previous step.
    energy_change
        The difference in energy between the current and previous step.
    """

    logdensity: float
    kinetic_change: float
    energy_change: float


def init(position: ArrayLike, logdensity_fn, rng_key):
    
    l, g = jax.value_and_grad(logdensity_fn)(position)

    return IntegratorState(
        position=position,
        momentum=jax.random.normal(rng_key, shape = (position.shape[0], )),
        logdensity=l,
        logdensity_grad=g,
    )


def build_kernel(logdensity_fn, sqrt_diag_cov, integrator):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the UnderdampedLangevin dynamics.
    L
        the momentum decoherence rate.
    step_size
        step size of the integrator.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    inverse_mass_matrix = sqrt_diag_cov # TODO: off by inverse?
    metric = metrics.default_metric(inverse_mass_matrix)
    # jax.debug.print("metric of 2 {x}", x=metric.kinetic_energy(jax.numpy.ones((10,))+1))
    step = with_maruyama(
        integrator(logdensity_fn=logdensity_fn, kinetic_energy_fn=metric.kinetic_energy)
    )

    def kernel(
        rng_key: PRNGKey, state: IntegratorState, L: float, step_size: float
    ) -> tuple[IntegratorState, UnderdampedLangevinInfo]:
        (position, momentum, logdensity, logdensitygrad) = step(
            state, step_size, L, rng_key
        )

        kinetic_change =  metric.kinetic_energy(momentum) - metric.kinetic_energy(state.momentum)

        return IntegratorState(
            position, momentum, logdensity, logdensitygrad
        ), UnderdampedLangevinInfo(
            logdensity=logdensity,
            energy_change=kinetic_change - logdensity + state.logdensity,
            kinetic_change=kinetic_change,
        )

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    L,
    step_size,
    integrator=velocity_verlet,
    sqrt_diag_cov=1.0,
) -> SamplingAlgorithm:
    """The general UnderdampedLangevin kernel builder (:meth:`blackjax.mcmc.underdamped_langevin.build_kernel`, alias `blackjax.langevin.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.langevin` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new underdamped_langevin kernel can be initialized and used with the following code:

    .. code::

        underdamped_langevin = blackjax.mcmc.underdamped_langevin.underdamped_langevin(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size
        )
        state = underdamped_langevin.init(position)
        new_state, info = underdamped_langevin.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

        step = jax.jit(underdamped_langevin.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    L
        the momentum decoherence rate
    step_size
        step size of the integrator
    integrator
        an integrator. We recommend using the default here.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel = build_kernel(logdensity_fn, sqrt_diag_cov, integrator)

    def init_fn(position: ArrayLike, rng_key: PRNGKey):
        return init(position, logdensity_fn, rng_key)

    def update_fn(rng_key, state):
        return kernel(rng_key, state, L, step_size)

    return SamplingAlgorithm(init_fn, update_fn)
