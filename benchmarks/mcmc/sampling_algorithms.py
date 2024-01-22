

import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
import blackjax

__all__ = ["samplers"]




# def run_hmc(
#     rng_key, initial_states, tuned_params, logdensity_fn, num_samples):
    
#     warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)

#     # we use 4 chains for sampling
#     rng_key, init_key, warmup_key = jax.random.split(rng_key, 3)
#     # init_keys = jax.random.split(init_key, num_chains)
#     # init_params = jax.vmap(init_param_fn)(init_keys)

#     (initial_states, tuned_params), _ = warmup.run(warmup_key, init_params, 1000)


#     states, infos = run_inference_algorithm(
#         rng_key=rng_key,
#         initial_state_or_position=initial_states,
#         inference_algorithm=blackjax.nuts,
#         num_steps=num_samples,
#         transform=lambda x: x.position,
#     )

def run_mclmc(logdensity_fn, num_steps, initial_position, key):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    kernel = blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
    )

    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
    )

    _, samples, _ = run_inference_algorithm(
        rng_key=run_key,
        initial_state_or_position=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda x: x.position,
    )

    return samples



samplers = {'mclmc' : run_mclmc} # , 'mhmchmc': run_mhmchmc}
