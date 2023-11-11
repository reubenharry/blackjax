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

import blackjax.mcmc.integrators as integrators
from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["MCLMCState", "MCLMCInfo", "init", "build_kernel", "mclmc", "Parameters"]


class Parameters(NamedTuple):
    """Tunable parameters"""

    L: float
    step_size: float
    inverse_mass_matrix: Array


MCLMCState = integrators.IntegratorState


class MCLMCInfo(NamedTuple):
    """Additional information on the MCLMC transition.

    This additional information can be used for debugging or computing
    diagnostics.
    """

    transformed_x: Array
    l: Array
    de: float


def init(x_initial: ArrayTree, logdensity_fn, random_key):
    grad_logp = jax.value_and_grad(logdensity_fn)
    l, g = grad_logp(x_initial)

    u = random_unit_vector(random_key, d=x_initial.shape[0])

    return MCLMCState(x_initial, u, l, g)


def random_unit_vector(random_key, d):
    u = jax.random.normal(random_key, shape=(d,))
    u /= jnp.sqrt(jnp.sum(jnp.square(u)))
    return u


def update_position(grad_logp):
    def update(step_size, x, u):
        xx = x + step_size * u
        ll, gg = grad_logp(xx)
        return xx, ll, gg

    return update



def update_momentum(step_size, u, g):
    """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
    similar to the implementation: https://github.com/gregversteeg/esh_dynamics
    There are no exponentials e^delta, which prevents overflows when the gradient norm is large.
    """
    g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
    e = g / g_norm
    ue = jnp.dot(u, e)
    dim = u.shape[0]
    delta = step_size * g_norm / (dim - 1)
    zeta = jnp.exp(-delta)
    uu = e * (1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta * u
    delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1 - ue) * zeta**2)
    return uu / jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r



def partially_refresh_momentum(u, random_key, nu):
    """Adds a small noise to u and normalizes."""
    z = nu * jax.random.normal(random_key, shape=(u.shape[0],))

    return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z)))


def update(hamiltonian_dynamics, partially_refresh_momentum, d):
    def step(x, u, g, random_key, L, step_size, inverse_mass_matrix):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change = hamiltonian_dynamics(
            x=x, u=u, g=g, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix
        )

        # Langevin-like noise
        nu = jnp.sqrt((jnp.exp(2 * step_size / L) - 1.0) / d)
        uu = partially_refresh_momentum(u=uu, random_key=random_key, nu=nu)

        return xx, uu, ll, gg, kinetic_change

    return step


def build_kernel(grad_logp, d, integrator, transform):
    hamiltonian_step = integrator(
        T=update_position(grad_logp), V=update_momentum, d=d
    )
    move = update(hamiltonian_step, partially_refresh_momentum, d)

    def kernel(
        rng_key: PRNGKey, state: MCLMCState, params: Parameters
    ) -> tuple[MCLMCState, MCLMCInfo]:
        x, u, l, g = state

        L, step_size, inverse_mass_matrix = params

        xx, uu, ll, gg, kinetic_change = move(
            x, u, g, rng_key, L, step_size, inverse_mass_matrix
        )
        de = kinetic_change + ll - l
        return MCLMCState(xx, uu, ll, gg), MCLMCInfo(transform(xx), ll, de)

    return kernel


lambda_c = 0.1931833275037836  # critical value of the lambda parameter for the minimal norm integrator


def minimal_norm(d, T, V):
    def step(x, u, g, step_size, inverse_mass_matrix):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V
        uu, r1 = V(step_size * lambda_c, u, g * inverse_mass_matrix)
        xx, ll, gg = T(step_size, x, 0.5 * uu * inverse_mass_matrix)
        uu, r2 = V(step_size * (1 - 2 * lambda_c), uu, gg * inverse_mass_matrix)
        xx, ll, gg = T(step_size, xx, 0.5 * uu * inverse_mass_matrix)
        uu, r3 = V(step_size * lambda_c, uu, gg * inverse_mass_matrix)

        # kinetic energy change
        kinetic_change = (r1 + r2 + r3) * (d - 1)

        return xx, uu, ll, gg, kinetic_change

    return step


class mclmc:
    """todo: add documentation"""

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        d: int,
        transform: Callable,
        integrator=minimal_norm,
    ) -> SamplingAlgorithm:
        grad_logp = jax.value_and_grad(logdensity_fn)

        kernel = cls.build_kernel(grad_logp, d, integrator, transform)

        def init_fn(position: ArrayLikeTree, rng_key: PRNGKey):
            return cls.init(position, logdensity_fn, rng_key)

        return SamplingAlgorithm(init_fn, kernel)
