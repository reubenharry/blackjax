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
"""Algorithms to adapt the MCLMC kernel parameters, namely step size and L."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from blackjax.adaptation.step_size import DualAveragingAdaptationState, dual_averaging_adaptation

from blackjax.diagnostics import effective_sample_size
from blackjax.mcmc.mhmclmc import rescale
from blackjax.util import pytree_size


class MCLMCAdaptationState(NamedTuple):
    """Represents the tunable parameters for MCLMC adaptation.

    L
        The momentum decoherent rate for the MCLMC algorithm.
    step_size
        The step size used for the MCLMC algorithm.
    """

    L: float
    step_size: float

def streaming_average(O, x, streaming_avg, weight, zero_prevention):
    """streaming average of f(x)"""
    total, average = streaming_avg
    average = (total * average + weight * O(x)) / (total + weight + zero_prevention)
    total += weight
    streaming_avg = (total, average)
    return streaming_avg

def mclmc_find_L_and_step_size(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
):
    """
    Finds the optimal value of the parameters for the MCLMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function used for the MCMC algorithm.
    num_steps
        The number of MCMC steps that will subsequently be run, after tuning.
    state
        The initial state of the MCMC algorithm.
    rng_key
        The random number generator key.
    frac_tune1
        The fraction of tuning for the first step of the adaptation.
    frac_tune2
        The fraction of tuning for the second step of the adaptation.
    frac_tune3
        The fraction of tuning for the third step of the adaptation.
    desired_energy_va
        The desired energy variance for the MCMC algorithm.
    trust_in_estimate
        The trust in the estimate of optimal stepsize.
    num_effective_samples
        The number of effective samples for the MCMC algorithm.

    Returns
    -------
    A tuple containing the final state of the MCMC algorithm and the final hyperparameters.
    """
    dim = pytree_size(state.position)
    params = MCLMCAdaptationState(jnp.sqrt(dim), jnp.sqrt(dim) * 0.25)
    part1_key, part2_key = jax.random.split(rng_key, 2)

    state, params = make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        desired_energy_var=desired_energy_var,
        trust_in_estimate=trust_in_estimate,
        num_effective_samples=num_effective_samples,
    )(state, params, num_steps, part1_key)

    if frac_tune3 != 0:
        state, params = make_adaptation_L(mclmc_kernel, frac=frac_tune3, Lfactor=0.4)(
            state, params, num_steps, part2_key
        )

    return state, params


def make_L_step_size_adaptation(
    kernel,
    dim,
    frac_tune1,
    frac_tune2,
    desired_energy_var=1e-3,
    trust_in_estimate=1.5,
    num_effective_samples=150,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for the unadjusted MCLMC"""

    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def predictor(previous_state, params, adaptive_state, rng_key):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
        Designed for the unadjusted MCHMC"""

        time, x_average, step_size_max = adaptive_state

        # dynamics
        next_state, info = kernel(
            rng_key=rng_key,
            state=previous_state,
            L=params.L,
            step_size=params.step_size,
        )
        # step updating
        success, state, step_size_max, energy_change = handle_nans(
            previous_state,
            next_state,
            params.step_size,
            step_size_max,
            info.energy_change,
        )

        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (
            jnp.square(energy_change) / (dim * desired_energy_var)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        params_new = params._replace(step_size=step_size)

        adaptive_state = (time, x_average, step_size_max)

        return state, params_new, adaptive_state, success


    def step(iteration_state, weight_and_key):
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""

        mask, rng_key = weight_and_key
        state, params, adaptive_state, streaming_avg = iteration_state

        state, params, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )

        # update the running average of x, x^2
        streaming_avg = streaming_average(
            O=lambda x: jnp.array([x, jnp.square(x)]),
            x=ravel_pytree(state.position)[0],
            streaming_avg=streaming_avg,
            weight=(1-mask)*success*params.step_size,
            zero_prevention=mask,
        )

        return (state, params, adaptive_state, streaming_avg), None

    def L_step_size_adaptation(state, params, num_steps, rng_key):
        num_steps1, num_steps2 = int(num_steps * frac_tune1), int(
            num_steps * frac_tune2
        )
        L_step_size_adaptation_keys = jax.random.split(rng_key, num_steps1 + num_steps2)

        # we use the last num_steps2 to compute the diagonal preconditioner
        mask = 1-jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        # initial state of the kalman filter

        # run the steps
        kalman_state, *_ = jax.lax.scan(
            step,
            init=(state, params, adap0, kalman_state),
            xs=(outer_weights, L_step_size_adaptation_keys),
            length=num_steps1 + num_steps2,
        )
        state, params, _, kalman_state_output = kalman_state

        L = params.L
        # determine L
        if num_steps2 != 0.0:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)
            L = jnp.sqrt(jnp.sum(variances))

        return state, MCLMCAdaptationState(L, params.step_size)

    return L_step_size_adaptation


def make_adaptation_L(kernel, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps = int(num_steps * frac)
        adaptation_L_keys = jax.random.split(key, num_steps)

        def step(state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                L=params.L,
                step_size=params.step_size,
            )
            return next_state, next_state.position

        state, samples = jax.lax.scan(
            f=step,
            init=state,
            xs=adaptation_L_keys,
        )

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        ess = effective_sample_size(flat_samples[None, ...])

        return state, params._replace(
            L=Lfactor * params.step_size * jnp.mean(num_steps / ess)
        )

    return adaptation_L


def handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change):
    """if there are nans, let's reduce the stepsize, and not update the state. The
    function returns the old state in this case."""

    reduced_step_size = 0.8
    p, unravel_fn = ravel_pytree(next_state.position)
    nonans = jnp.all(jnp.isfinite(p))
    state, step_size, kinetic_change = jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (next_state, step_size_max, kinetic_change),
        (previous_state, step_size * reduced_step_size, 0.0),
    )

    return nonans, state, step_size, kinetic_change




Lratio_lowerbound = 0.5
Lratio_upperbound = 1 / Lratio_lowerbound


def mhmclmc_find_L_and_step_size(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    params=None
):
    """
    Finds the optimal value of the parameters for the MH-MCHMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function used for the MCMC algorithm.
    num_steps
        The number of MCMC steps that will subsequently be run, after tuning.
    state
        The initial state of the MCMC algorithm.
    rng_key
        The random number generator key.
    frac_tune1
        The fraction of tuning for the first step of the adaptation.
    frac_tune2
        The fraction of tuning for the second step of the adaptation.
    frac_tune3
        The fraction of tuning for the third step of the adaptation.
    desired_energy_va
        The desired energy variance for the MCMC algorithm.
    trust_in_estimate
        The trust in the estimate of optimal stepsize.
    num_effective_samples
        The number of effective samples for the MCMC algorithm.

    Returns
    -------
    A tuple containing the final state of the MCMC algorithm and the final hyperparameters.
    """

    dim = pytree_size(state.position)
    if params is None:
        params = MCLMCAdaptationState(jnp.sqrt(dim), jnp.sqrt(dim) * 0.1)
    else:
        params = params
    jax.debug.print("initial params {x}", x=params)
    part1_key, part2_key = jax.random.split(rng_key, 2)

    state, params = mhmclmc_make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
    )(state, params, num_steps, part1_key)

    if frac_tune3 != 0:
        
        part2_key1, part2_key2 = jax.random.split(part2_key, 2)

        state, params = mhmclmc_make_adaptation_L(mclmc_kernel, frac=frac_tune3, Lfactor=0.4)(
            state, params, num_steps, part2_key1
        )

        state, params = mhmclmc_make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=0,
        fix_L_first_da=True,
        )(state, params, num_steps, part2_key2)

    return state, params


def mhmclmc_make_L_step_size_adaptation(
    kernel,
    dim,
    frac_tune1,
    frac_tune2,
    fix_L_first_da=False,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for the unadjusted MCLMC"""


    

    def dual_avg_step(fix_L, update_da):
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""

        def step(iteration_state, weight_and_key):

            mask, rng_key = weight_and_key
            kernel_key, num_steps_key = jax.random.split(rng_key, 2)
            previous_state, params, (adaptive_state, step_size_max), streaming_avg = iteration_state


            # num_integration_steps = jnp.min(jnp.array([jax.random.poisson(num_steps_key, params.L/params.step_size)+1, 1000]))
            # num_integration_steps = jnp.min(jnp.array([jax.random.poisson(num_steps_key, params.L/params.step_size), 1000]))
            # num_integration_steps = jnp.round(jax.random.uniform(num_steps_key) * rescale(params.L/params.step_size + 0.5))
            avg_num_integration_steps = params.L/params.step_size

            # jax.debug.print("{x} avg_",x=(params.L/params.step_size, params))

            # dynamics
            state, info = kernel(
                rng_key=kernel_key,
                state=previous_state,
                avg_num_integration_steps=avg_num_integration_steps,
                step_size=params.step_size,
            )

            # step updating
            success, state, step_size_max, energy_change = handle_nans(
                previous_state,
                state,
                params.step_size,
                step_size_max,
                info.energy,
            )

            # jax.debug.print("info acc rate {x}", x=(info,))
            # jax.debug.print("state {x}", x=(state.position,))


            log_step_size, log_step_size_avg, step, avg_error, mu = update_da(
                adaptive_state, info.acceptance_rate)
            
            adaptive_state = DualAveragingAdaptationState(
                mask * log_step_size + (1-mask)*adaptive_state.log_step_size, 
                mask * log_step_size_avg + (1-mask)*adaptive_state.log_step_size_avg,
                mask * step + (1-mask)*adaptive_state.step,
                mask * avg_error + (1-mask)*adaptive_state.avg_error,
                mask * mu + (1-mask)*adaptive_state.mu,
                )

            # jax.debug.print("{x} step_size before",x=(adaptive_state.log_step_size, info.acceptance_rate,))
            # adaptive_state = update(adaptive_state, info.acceptance_rate)
            # jax.debug.print("{x} step_size after",x=(adaptive_state.log_step_size,))
            

            # step_size = jax.lax.clamp(1e-3, jnp.exp(adaptive_state.log_step_size), 1e0)
            # step_size = jax.lax.clamp(1e-5, jnp.exp(adaptive_state.log_step_size), step_size_max)
            step_size = jax.lax.clamp(1e-5, jnp.exp(adaptive_state.log_step_size), params.L)
            # step_size = 1e-3

            # update the running average of x, x^2
            streaming_avg = streaming_average(
                O=lambda x: jnp.array([x, jnp.square(x)]),
                x=ravel_pytree(state.position)[0],
                streaming_avg=streaming_avg,
                weight=(1-mask)*success*step_size,
                zero_prevention=mask,
            )

            # n = L/eps
            # eps -> eps * new_eps/eps 

            # params = params._replace(step_size=step_size)
            # jax.debug.print("new step size {x}", x=step_size)

            # params = params._replace(
            #         step_size=step_size, 
            #         L = params.L * (step_size / params.step_size)
            #         )
            
            if fix_L:
                params = params._replace(
                        step_size=mask * step_size + (1-mask)*params.step_size, 

                        # L=mask * ((params.L * (step_size / params.step_size))) + (1-mask)*params.L
                        # L=mask * ((params.L * (step_size / params.step_size))) + (1-mask)*params.L
                        )
            
            else:

                params = params._replace(
                        step_size=mask * step_size + (1-mask)*params.step_size, 

                        L=mask * ((params.L * (step_size / params.step_size))) + (1-mask)*params.L
                        # L=mask * ((params.L * (step_size / params.step_size))) + (1-mask)*params.L
                        )
            # params = params._replace(step_size=step_size, 
            #                         L=(params.L/params.step_size * step_size)
            #                         )


            return (state, params, (adaptive_state, step_size_max), streaming_avg), info
        return step
    
    def step_size_adaptation(mask, state, params, keys, fix_L, initial_da, update_da):
    
        return jax.lax.scan(
            dual_avg_step(fix_L, update_da),
            init=(
                state,
                params,
                (initial_da(params.step_size), jnp.inf), # step size max
                # (init(params.step_size), params.L/4),
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=(mask, keys),
    )

    def L_step_size_adaptation(state, params, num_steps, rng_key):
        num_steps1, num_steps2 = int(num_steps * frac_tune1), int(
            num_steps * frac_tune2
        )

        rng_key_pass1, rng_key_pass2 = jax.random.split(rng_key, 2)
        L_step_size_adaptation_keys_pass1 = jax.random.split(rng_key_pass1, num_steps1 + num_steps2)
        L_step_size_adaptation_keys_pass2 = jax.random.split(rng_key_pass2, num_steps1 + num_steps2)

        # determine which steps to ignore in the streaming average
        mask = 1 - jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        initial_da, update_da, final_da = dual_averaging_adaptation(target=0.65)

        ((state, params, (dual_avg_state, step_size_max), (_, average)), info) = step_size_adaptation(mask, state, params, L_step_size_adaptation_keys_pass1, fix_L=fix_L_first_da, initial_da=initial_da, update_da=update_da)
        params = params._replace(step_size=final_da(dual_avg_state))
        

        jax.debug.print("{x}",x=("mean acceptance rate", jnp.mean(info.acceptance_rate,)))
        jax.debug.print("{x} params after a round of tuning",x=(params))
        # jax.debug.print("{x} step size max",x=(step_size_max))
        # jax.debug.print("{x} final",x=(final(dual_avg_state)))

        # raise Exception

        # determine L
        if num_steps2 != 0.0:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)
            change = jax.lax.clamp(Lratio_lowerbound, jnp.sqrt(jnp.sum(variances))/params.L, Lratio_upperbound)
            jax.debug.print("{x} L ratio, old val,  new val",x=(change, params.L, params.L*change))
            params = params._replace(L=params.L*change)

        
            jax.debug.print("{x} params before second round",x=(params))
            initial_da, update_da, final_da = dual_averaging_adaptation(target=0.65)
            ((state, params, (dual_avg_state, step_size_max), (_, average)), info) = step_size_adaptation(mask, state, params, L_step_size_adaptation_keys_pass2, fix_L=True, update_da=update_da, initial_da=initial_da)
            params = params._replace(step_size=final_da(dual_avg_state))
            jax.debug.print("{x}",x=("mean acceptance rate", jnp.mean(info.acceptance_rate,)))
            jax.debug.print("{x} params after a round of tuning",x=(params))

        return state, params
    
    return L_step_size_adaptation



def mhmclmc_make_adaptation_L(kernel, frac, Lfactor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps = int(num_steps * frac)
        adaptation_L_keys = jax.random.split(key, num_steps)

        
        def step(state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                step_size=params.step_size,
                avg_num_integration_steps=params.L/params.step_size,
            )
            return next_state, next_state.position

        state, samples = jax.lax.scan(
            f=step,
            init=state,
            xs=adaptation_L_keys,
        )


        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        ess = effective_sample_size(flat_samples[None, ...])

        change = jax.lax.clamp(Lratio_lowerbound, (Lfactor * params.step_size * jnp.mean(num_steps / ess))/params.L, Lratio_upperbound)

        jax.debug.print("tune 3\n\n {x}", x=(params.L*change, change))
        return state, params._replace(
            # L=Lfactor * params.step_size * jnp.mean(num_steps / ess)
            L=params.L*change
        )

    return adaptation_L










