import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import optax
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState, handle_nans
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.diagnostics import effective_sample_size
from blackjax.mcmc.adjusted_mclmc import rescale
from blackjax.util import pytree_size, incremental_value_update, run_inference_algorithm

from blackjax.mcmc.integrators import (
    generate_euclidean_integrator,
    generate_isokinetic_integrator,
    mclachlan,
    yoshida,
    velocity_verlet,
    omelyan,
    isokinetic_mclachlan,
    isokinetic_velocity_verlet,
    isokinetic_yoshida,
    isokinetic_omelyan,
)

Lratio_lowerbound = 0.0
Lratio_upperbound = 2.0


def adjusted_mclmc_find_L_and_step_size(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    target,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    diagonal_preconditioning=True,
    params=None,
    max=False,
    num_windows=1,
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
    target
        The target acceptance rate for the step size adaptation.
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
        params = MCLMCAdaptationState(
            jnp.sqrt(dim), jnp.sqrt(dim) * 0.2, sqrt_diag_cov=jnp.ones((dim,))
        )

    part1_key, part2_key = jax.random.split(rng_key, 2)

    for _ in range(num_windows):
        part1_key = jax.random.split(part1_key)[0]
        (
            state,
            params,
            params_history,
            final_da_val,
        ) = adjusted_mclmc_make_L_step_size_adaptation(
            kernel=mclmc_kernel,
            dim=dim,
            frac_tune1=frac_tune1,
            frac_tune2=frac_tune2,
            target=target,
            diagonal_preconditioning=diagonal_preconditioning,
            max=max,
        )(
            state, params, num_steps, part1_key
        )
    # jax.debug.print("params after stage 2 {x}", x=params)

    if frac_tune3 != 0:
        part2_key1, part2_key2 = jax.random.split(part2_key, 2)

        state, params = adjusted_mclmc_make_adaptation_L(
            mclmc_kernel, frac=frac_tune3, Lfactor=0.4, max=max
        )(state, params, num_steps, part2_key1)

        # jax.debug.print("params after stage 3 {x}", x=params)
        (
            state,
            params,
            params_history,
            final_da_val,
        ) = adjusted_mclmc_make_L_step_size_adaptation(
            kernel=mclmc_kernel,
            dim=dim,
            frac_tune1=frac_tune1,
            frac_tune2=0,
            target=target,
            fix_L_first_da=True,
            diagonal_preconditioning=diagonal_preconditioning,
            max=max,
        )(
            state, params, num_steps, part2_key2
        )

       
        # jax.debug.print("params after stage 1 (again) {x}", x=params)
        

    return state, params, params_history, final_da_val


def adjusted_mclmc_make_L_step_size_adaptation(
    kernel,
    dim,
    frac_tune1,
    frac_tune2,
    target,
    diagonal_preconditioning,
    fix_L_first_da=False,
    max=False,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for the unadjusted MCLMC"""

    def dual_avg_step(fix_L, update_da):
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""

        def step(iteration_state, weight_and_key):


            mask, rng_key = weight_and_key
            # kernel_key, num_steps_key = jax.random.split(rng_key, 2)
            rng_key, nan_key = jax.random.split(rng_key, 2)
            (
                previous_state,
                params,
                (adaptive_state, step_size_max),
                previous_weight_and_average,
            ) = iteration_state

            avg_num_integration_steps = params.L / params.step_size

            state, info = kernel(
                rng_key=rng_key,
                state=previous_state,
                avg_num_integration_steps=avg_num_integration_steps,
                step_size=params.step_size,
                sqrt_diag_cov=params.sqrt_diag_cov,
            )
            # jax.debug.print("{x}", x=(params.step_size))
            # jax.debug.print(" acc rate {x}", x=(info.acceptance_rate))

            # jax.debug.print("acceptance rate {x}", x=info.acceptance_rate)
            # jax.debug.print("{x}", x=params.step_size)
            # jax.debug.print("L {x}", x=params.L)

            # step updating
            success, state, step_size_max, energy_change = handle_nans(
                previous_state,
                state,
                params.step_size,
                step_size_max,
                info.energy,
                nan_key
            )

            log_step_size, log_step_size_avg, step, avg_error, mu = update_da(
                adaptive_state, info.acceptance_rate
            )

            adaptive_state = DualAveragingAdaptationState(
                mask * log_step_size + (1 - mask) * adaptive_state.log_step_size,
                mask * log_step_size_avg
                + (1 - mask) * adaptive_state.log_step_size_avg,
                mask * step + (1 - mask) * adaptive_state.step,
                mask * avg_error + (1 - mask) * adaptive_state.avg_error,
                mask * mu + (1 - mask) * adaptive_state.mu,
            )

            step_size = jax.lax.clamp(
                1e-5, jnp.exp(adaptive_state.log_step_size), params.L / 1.1
            )
            adaptive_state = adaptive_state._replace(log_step_size=jnp.log(step_size))
            # step_size = 1e-3

            x = ravel_pytree(state.position)[0]
            # update the running average of x, x^2
            previous_weight_and_average = incremental_value_update(
                expectation=jnp.array([x, jnp.square(x)]),
                incremental_val=previous_weight_and_average,
                weight=(1 - mask) * success * step_size,
                zero_prevention=mask,
            )

            if fix_L:
                params = params._replace(
                    step_size=mask * step_size + (1 - mask) * params.step_size,
                )

            else:
                params = params._replace(
                    step_size=mask * step_size + (1 - mask) * params.step_size,
                    L=mask * (params.L * (step_size / params.step_size))
                    + (1 - mask) * params.L,
                )

            return (
                state,
                params,
                (adaptive_state, step_size_max),
                previous_weight_and_average,
            ), (
                info,
                params,
            )

        return step

    def step_size_adaptation(mask, state, params, keys, fix_L, initial_da, update_da):
        return jax.lax.scan(
            dual_avg_step(fix_L, update_da),
            init=(
                state,
                params,
                (initial_da(params.step_size), jnp.inf),  # step size max
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=(mask, keys),
        )
    
    def adam_step(loss_fn, optimizer, opt_state, previous_state, params, fix_L, rng_key):
        # use optax to vary stepsize, to minimize the difference between the acceptance rate and the target acceptance rate


        # loss_fn(step_size=params.step_size, state=previous_state, key=rng_key)
        
        # grads = jax.grad(loss_fn)(params.step_size, previous_state, rng_key)

        # jax.debug.print("loss {x}", x=(loss_fn(params, state, keys[0])))


        # todo: should i use different key here? seems slightly subtle
        new_state, info = kernel(
            rng_key=rng_key,
            state=previous_state,
            step_size=params.step_size,
            avg_num_integration_steps=params.L / params.step_size,
            sqrt_diag_cov=params.sqrt_diag_cov,
        )

        grads = target - info.acceptance_rate
        updates, opt_state = optimizer.update(grads, opt_state)
        new_step_size = optax.apply_updates(params.step_size, updates)

        # jax.debug.print("stepsize {x}", x=(params.step_size,new_step_size))
        # new_step_size = params.step_size

        mask = 1.0

        if fix_L:
                params = params._replace(
                    step_size=mask*new_step_size + (1 - mask) * params.step_size,
                )

        else:
            params = params._replace(
                step_size=mask * new_step_size + (1 - mask) * params.step_size,
                L=mask * (params.L * (new_step_size / params.step_size))
                + (1 - mask) * params.L,
            )

        # jax.debug.print("stepsize after {x}", x=(params.step_size, new_step_size))

        return (params, new_state, opt_state), info


    
    def adam_adaptation(state, params, target_acceptance_rate, fix_L, keys):

        start_learning_rate = 1e-1
        optimizer = optax.adam(learning_rate=start_learning_rate)
        opt_state = optimizer.init(params.step_size)

        # def loss_fn(step_size, state, key):
            # return 5.0
            # _, info = kernel(
            #         rng_key=key,
            #         state=state,
            #         step_size=step_size,
            #         avg_num_integration_steps=params.L / step_size,
            #         sqrt_diag_cov=params.sqrt_diag_cov,
            #     )
            
            # return jnp.square(info.acceptance_rate.mean() - target_acceptance_rate)
        
        # adam_step(loss_fn, optimizer, opt_state, state, params, fix_L, keys[0])
    
        return jax.lax.scan(
            lambda vars, key: adam_step(loss_fn=None, optimizer=optimizer, fix_L=fix_L, rng_key=key, opt_state=vars[2], previous_state=vars[1], params=vars[0]),
            (params, state, opt_state),
            keys
        )

        

    def L_step_size_adaptation(state, params, num_steps, rng_key):
        num_steps1, num_steps2 = int(num_steps * frac_tune1), int(
            num_steps * frac_tune2
        )

        check_key, rng_key = jax.random.split(rng_key, 2)

        rng_key_pass1, rng_key_pass2 = jax.random.split(rng_key, 2)
        L_step_size_adaptation_keys_pass1 = jax.random.split(
            rng_key_pass1, num_steps1 + num_steps2
        )
        L_step_size_adaptation_keys_pass2 = jax.random.split(rng_key_pass2, num_steps1)

        # determine which steps to ignore in the streaming average
        mask = 1 - jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        initial_da, update_da, final_da = dual_averaging_adaptation(target=target)

        (
            (state, params, (dual_avg_state, step_size_max), (_, average)),
            (info, params_history),
        ) = step_size_adaptation(
            mask,
            state,
            params,
            L_step_size_adaptation_keys_pass1,
            fix_L=fix_L_first_da,
            initial_da=initial_da,
            update_da=update_da,
        )


        final_stepsize = final_da(dual_avg_state)
        params = params._replace(step_size=final_stepsize)

        # jax.debug.print("stepsize {x}", x=params.step_size)
        # jax.debug.print("L {x}", x=params.L)
        # jax.debug.print("info {x}", x=info.acceptance_rate.mean())

        # L_step_size_adaptation_keys_pass1 = jax.random.split(check_key, num_steps1)
        # mask = jnp.zeros(num_steps1)

        # (
        #     (state, params, (dual_avg_state, step_size_max), (_, average)),
        #     (info, params_history),
        # ) = step_size_adaptation(
        #     mask,
        #     state,
        #     params,
        #     L_step_size_adaptation_keys_pass1,
        #     fix_L=fix_L_first_da,
        #     initial_da=initial_da,
        #     update_da=update_da,
        # )

        # keys = jax.random.split(check_key, 1000)
        # (params,state,opt_state), info = adam_adaptation(state, params, target, fix_L_first_da, keys)
        # jax.debug.print("stepsize {x}", x=params.step_size)
        # # raise Exception

        # keys = jax.random.split(check_key, 1000)
        # _, info = jax.lax.scan(
        #     lambda state, key: kernel(
        #         rng_key=key,
        #         state=state,
        #         # step_size=params.step_size,
        #         step_size=final_stepsize,
        #         avg_num_integration_steps=params.L / params.step_size,
        #         sqrt_diag_cov=params.sqrt_diag_cov,
        #     ),
        #     state,
        #     keys
        # )

        # jax.debug.print("pst run {x}", x=info.acceptance_rate.mean())
        # jax.debug.print("cov {x}", x=params.sqrt_diag_cov)
        # raise Exception

        # num_steps_per_traj = params.L / params.step_size
        # alg = blackjax.adjusted_mclmc(
        #     logdensity_fn=lambda x : -0.5 * jnp.sum(jnp.square(x), axis= -1),
        #     step_size=params.step_size,
        #     integration_steps_fn=lambda k: jnp.ceil(jax.random.uniform(k) * rescale(num_steps_per_traj)),
        #     integrator= isokinetic_mclachlan,
        #     sqrt_diag_cov=1.0,
        #     L_proposal_factor=jnp.inf,
        # )
        # _, (_, info) = run_inference_algorithm(
        #     rng_key=jax.random.PRNGKey(0),
        #     inference_algorithm=alg,
        #     initial_state=state,
        #     num_steps=2000 
        # )

        # jax.debug.print("pst run {x}", x=info.acceptance_rate.mean())
        # raise Exception



        # determine L
        if num_steps2 != 0.0:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)

            if max:
                contract = lambda x: jnp.max(x)*dim
            else:
                contract = jnp.sum

            change = jax.lax.clamp(
                Lratio_lowerbound,
                jnp.sqrt(contract(variances)) / params.L,
                Lratio_upperbound,
            )
            params = params._replace(
                L=params.L * change, step_size=params.step_size * change
            )
            if diagonal_preconditioning:
                params = params._replace(
                    sqrt_diag_cov=jnp.sqrt(variances), L=jnp.sqrt(dim)
                )

            initial_da, update_da, final_da = dual_averaging_adaptation(target=target)
            (
                (state, params, (dual_avg_state, step_size_max), (_, average)),
                (info, params_history),
            ) = step_size_adaptation(
                jnp.ones(num_steps1),
                state,
                params,
                L_step_size_adaptation_keys_pass2,
                fix_L=True,
                update_da=update_da,
                initial_da=initial_da,
            )

            params = params._replace(step_size=final_da(dual_avg_state))

            # jax.debug.print("info 2 {x}", x=info.acceptance_rate.mean())
            # keys = jax.random.split(check_key, 1000)
            # _, info = jax.lax.scan(
            #     lambda state, key: kernel(
            #         rng_key=key,
            #         state=state,
            #         # step_size=params.step_size,
            #         step_size=params.step_size,
            #         avg_num_integration_steps=params.L / params.step_size,
            #         sqrt_diag_cov=params.sqrt_diag_cov,
            #     ),
            #     state,
            #     keys
            # )

            # jax.debug.print("pst run {x}", x=info.acceptance_rate.mean())
            # jax.debug.print("stepsize and L {x}", x=(params.step_size, params.L))

        return state, params, params_history.step_size, final_da(dual_avg_state)

    return L_step_size_adaptation


def adjusted_mclmc_make_adaptation_L(kernel, frac, Lfactor, max=False):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps = int(num_steps * frac)
        adaptation_L_keys = jax.random.split(key, num_steps)

        def step(state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                step_size=params.step_size,
                avg_num_integration_steps=params.L / params.step_size,
                sqrt_diag_cov=params.sqrt_diag_cov,
            )
            return next_state, next_state.position

        state, samples = jax.lax.scan(
            f=step,
            init=state,
            xs=adaptation_L_keys,
        )

        if max:
            contract = jnp.min
        else:
            contract = jnp.mean

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        # number of effective samples per 1 actual sample
        ess = contract(effective_sample_size(flat_samples[None, ...]))/num_steps

        # jax.debug.print("{x}foo", x=jnp.mean(ess)/num_steps)

        return state, params._replace(L=Lfactor * params.L / jnp.mean(ess))

    return adaptation_L


def handle_nans(
    previous_state, next_state, step_size, step_size_max, kinetic_change, key
):
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
