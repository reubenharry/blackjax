

import jax
import jax.numpy as jnp
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState, integrator_order, target_acceptance_rate_of_order
from blackjax.mcmc.integrators import calls_per_integrator_step, generate_euclidean_integrator, generate_isokinetic_integrator
from blackjax.mcmc.mhmclmc import rescale
from blackjax.util import run_inference_algorithm
import blackjax

__all__ = ["samplers"]




def run_nuts(
    coefficients, logdensity_fn, num_steps, initial_position, transform, key):
    
    integrator = generate_euclidean_integrator(coefficients)
    # integrator = blackjax.mcmc.integrators.velocity_verlet # note: defaulted to in nuts
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn, integrator=integrator)

    # we use 4 chains for sampling
    rng_key, warmup_key = jax.random.split(key, 2)


    (state, params), _ = warmup.run(warmup_key, initial_position, 2000)

    nuts = blackjax.nuts(logdensity_fn=logdensity_fn, step_size=params['step_size'], inverse_mass_matrix= params['inverse_mass_matrix'], integrator=integrator)

    final_state, state_history, info_history = run_inference_algorithm(
        rng_key=rng_key,
        initial_state_or_position=state,
        inference_algorithm=nuts,
        num_steps=num_steps,
        transform=lambda x: transform(x.position),
    )

    # print("INFO\n\n",info_history.num_integration_steps)

    return state_history, params, info_history.num_integration_steps.mean() * calls_per_integrator_step(coefficients)

def run_mclmc(coefficients, logdensity_fn, num_steps, initial_position, transform, key):

    integrator = generate_isokinetic_integrator(coefficients)

    init_key, tune_key, run_key = jax.random.split(key, 3)


    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    # integrator = blackjax.mcmc.integrators.isokinetic_mclachlan
    
    kernel = lambda std_mat : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=integrator,
        # std_mat=jnp.ones((initial_position.shape[0],)),
        std_mat=std_mat,
    )

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        # frac_tune3=0.0,
        diagonal_preconditioning=False
    )

    # jax.debug.print("params {x}", x=(blackjax_mclmc_sampler_params.L, blackjax_mclmc_sampler_params.step_size))

    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
        std_mat=blackjax_mclmc_sampler_params.std_mat,
        integrator = integrator,

        # std_mat=jnp.ones((initial_position.shape[0],)),
    )

    _, samples, _ = run_inference_algorithm(
        rng_key=run_key,
        initial_state_or_position=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda x: transform(x.position),
        progress_bar=True,
    )

    return samples, blackjax_mclmc_sampler_params, calls_per_integrator_step(coefficients)


def run_mhmclmc(coefficients, logdensity_fn, num_steps, initial_position, transform, key):
    integrator = generate_isokinetic_integrator(coefficients)

    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_state = blackjax.mcmc.mhmclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, random_generator_arg=init_key
    )

    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, std_mat: blackjax.mcmc.mhmclmc.build_kernel(
                integrator=integrator,
                integration_steps_fn = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(avg_num_integration_steps)),
                std_mat=std_mat,
            )(
                rng_key=rng_key, 
                state=state, 
                step_size=step_size, 
                logdensity_fn=logdensity_fn)

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.adaptation.mclmc_adaptation.mhmclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=target_acceptance_rate_of_order[integrator_order(coefficients)],
        frac_tune1=0.1,
        frac_tune2=0.1,
        frac_tune3=0.1,
        diagonal_preconditioning=False,
    )

    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L
    # jax.debug.print("params {x}", x=(blackjax_mclmc_sampler_params.std_mat))


    alg = blackjax.mcmc.mhmclmc.mhmclmc(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn = lambda key: jnp.ceil(jax.random.uniform(key) * rescale(L/step_size)) ,
        integrator=integrator,
        std_mat=blackjax_mclmc_sampler_params.std_mat,
        

    )

    _, out, info = run_inference_algorithm(
        rng_key=run_key,
        initial_state_or_position=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps, 
        transform=lambda x: transform(x.position), 
        progress_bar=True)


    return out, blackjax_mclmc_sampler_params, calls_per_integrator_step(coefficients) * (L/step_size)

# we should do at least: mclmc, nuts, unadjusted hmc, mhmclmc, langevin

samplers = {
    'nuts' : run_nuts,
    'mclmc' : run_mclmc, 
    'mhmclmc': run_mhmclmc, 
    }


# foo = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(20.56))

# print(jnp.mean(jax.vmap(foo)(jax.random.split(jax.random.PRNGKey(1), 10000000))))