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
"""Public API for the Stochastic gradient Microcanonical Langevin Dynamics kernel."""
from typing import Callable

import blackjax.sgmcmc.diffusions as diffusions
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "build_kernel", "sgmclmc"]


def init(position: ArrayLikeTree) -> ArrayLikeTree:
    return position


def build_kernel() -> Callable:
    """Stochastic gradient Microcanonical Langevin Dynamics (SGMCLMC) algorithm."""
    integrator = diffusions.mclmc()

    def kernel(
        rng_key: PRNGKey,
        state: ArrayLikeTree,
        grad_estimator: Callable,
        minibatch: ArrayLikeTree,
        L : float,
        step_size: float,
    ) -> ArrayTree:
        position, _ = state
        logdensity_grad_fn = lambda pos : grad_estimator(pos, minibatch)

        return integrator(
            rng_key, state, logdensity_grad_fn, L, step_size, 
        )

    return kernel


class sgmclmc:
    """Implements the (basic) user interface for the SGMCLMC kernel.

    The general sgmclmc kernel builder (:meth:`blackjax.sgmcmc.sgmclmc.build_kernel`, alias
    `blackjax.sgmclmc.build_kernel`) can be cumbersome to manipulate. Since most users
    only need to specify the kernel parameters at initialization time, we
    provide a helper function that specializes the general kernel.

    """

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        grad_estimator: Callable,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel()

        def init_fn(position: ArrayLikeTree, rng_key=None):
            del rng_key
            return cls.init(position)

        def step_fn(
            rng_key: PRNGKey,
            state: ArrayLikeTree,
            minibatch: ArrayLikeTree,
            L : float,
            step_size: float,
        ) -> ArrayTree:
            return kernel(
                rng_key, state, grad_estimator, minibatch, L, step_size,
            )

        return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
