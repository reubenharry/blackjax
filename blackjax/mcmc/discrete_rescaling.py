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

from blackjax.mcmc import integrators
from blackjax.mcmc.mclmc import MCLMCInfo
import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import (
    IntegratorState,
    isokinetic_mclachlan,
    velocity_verlet,
    with_isokinetic_maruyama,
)
from blackjax.types import ArrayLike, PRNGKey
from blackjax.util import generate_unit_vector, pytree_size

__all__ = ["init", "build_kernel", "as_top_level_api"]




def init(position: ArrayLike, logdensity_fn, rng_key):

    l, g = jax.value_and_grad(logdensity_fn)(position)

    return IntegratorState(
        position=position,
        momentum=generate_unit_vector(rng_key, position),
        logdensity=l,
        logdensity_grad=g,
    )


def build_kernel(logdensity_fn, sqrt_diag_cov, integrator):
    """Build a HMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics.
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

    # step = with_isokinetic_maruyama(
    #     integrator(logdensity_fn=logdensity_fn, sqrt_diag_cov=sqrt_diag_cov)
    # )
    # kinetic_energy_fn = lambda p : 0.5 * jnp.dot(p, p)
    kinetic_energy_fn = lambda p : jnp.log(jnp.linalg.norm(p))
    step = integrator(logdensity_fn=logdensity_fn, kinetic_energy_fn=kinetic_energy_fn)

    def rescaled_step(state, step_size): 
        conformal_factor = jax.numpy.linalg.norm(state.momentum)
        step_size = step_size * conformal_factor
        return step(state, step_size)

    def kernel(
        rng_key: PRNGKey, state: IntegratorState, L: float, step_size: float
    ) -> tuple[IntegratorState, MCLMCInfo]:
        (position, momentum, logdensity, logdensitygrad) = rescaled_step(
            state, step_size
        )

        dim = pytree_size(position)
        
        nu = jnp.sqrt((jnp.exp(2*step_size / L) - 1)/dim)
        z = jax.random.normal(rng_key, shape=(dim,))

        momentum = momentum + (nu*z) / jnp.linalg.norm(nu*z)
        kinetic_change = kinetic_energy_fn(state.momentum) - kinetic_energy_fn(momentum)

        return IntegratorState(
            position, momentum, logdensity, logdensitygrad
        ), MCLMCInfo(
            logdensity=logdensity,
            energy_change= kinetic_change - logdensity + state.logdensity,
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
    """The general mclmc kernel builder (:meth:`blackjax.mcmc.mclmc.build_kernel`, alias `blackjax.mclmc.build_kernel`) can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.mclmc` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new mclmc kernel can be initialized and used with the following code:

    .. code::

        mclmc = blackjax.mcmc.mclmc.mclmc(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size
        )
        state = mclmc.init(position)
        new_state, info = mclmc.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

        step = jax.jit(mclmc.step)
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
