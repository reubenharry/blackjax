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
from typing import Callable, NamedTuple, Any
import jax
import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import IntegratorState, noneuclidean_mclachlan
from blackjax.types import ArrayLike, PRNGKey

from blackjax.mcmc import mclmc

#__all__ = ["MCLMCInfo", "init", "build_kernel", "mclmc"]


class Info(NamedTuple):
    x: Any #MCLMCInfo type
    y: Any


class State(NamedTuple):
    x: Any
    y: Any #TODO
    
    
def n_step_kernel(one_step, n):
    """do one_step n times"""
    
    def kernel(rng_key, state, L, stepsize, args):
        state, track = jax.lax.scan(lambda _state, key: one_step(key, _state, L, stepsize, args), init= state, xs = jax.random.split(rng_key, n))
        last_in_track = jax.tree_util.tree_map(lambda x: x[-1], track) # store only the info for the last step
        return state, last_in_track
    
    return kernel
    
    
def init(position_x: ArrayLike, position_y: ArrayLike, logdensity_fn, rng_key):
    
    key1, key2 = jax.random.split(rng_key)
    
    return State(mclmc.init(position_x, lambda x: logdensity_fn(x, position_y), key1), mclmc.init(position_y, lambda y: logdensity_fn(position_x, y), key2))
    


def build_kernel(logdensity_fn, integrator, nx, ny):
    """ Gibbs MCLMC kernel. We have the logdensity(x, y).
        First does nx steps with fixed y, then ny steps with fixed x.
    """
    
    kernel1 = n_step_kernel(mclmc.build_kernel(logdensity_fn, integrator), nx) # nx steps with fixed y
    kernel2 = n_step_kernel(mclmc.build_kernel(lambda y, xfixed: logdensity_fn(xfixed, y), integrator), ny) # ny steps with fixed x
    
    def gibbs_kernel(rng_key, state, hyp):
        L1, stepsize1, L2, stepsize2 = hyp
        key1, key2 = jax.random.split(rng_key)
        x_new, info_x = kernel1(key1, state.x, L1, stepsize1, state.y.position)
        y_new, info_y = kernel2(key2, state.y, L2, stepsize2, x_new.position)
        
        return State(x_new, y_new), Info(info_x, info_y)
        
    return gibbs_kernel




class mclmcGibbs:

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        hyp,
        integrator=noneuclidean_mclachlan,
        nx=1,
        ny=1
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(logdensity_fn, integrator, nx, ny)

        def init_fn(position: ArrayLike, rng_key: PRNGKey):
            return cls.init(position[0], position[1], logdensity_fn, rng_key)

        def update_fn(rng_key, state):
            return kernel(rng_key, state, hyp)

        return SamplingAlgorithm(init_fn, update_fn)
    