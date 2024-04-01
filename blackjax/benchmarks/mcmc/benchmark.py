from collections import defaultdict
from functools import partial
import math
import operator
import os
from statistics import mean, median
import jax
import jax.numpy as jnp
import pandas as pd
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState, integrator_order, target_acceptance_rate_of_order



# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(128)
# num_cores = jax.local_device_count()
# print(num_cores, jax.lib.xla_bridge.get_backend().platform)

import itertools

import numpy as np

import blackjax
from blackjax.benchmarks.mcmc.sampling_algorithms import samplers
from blackjax.benchmarks.mcmc.inference_models import MixedLogit, StandardNormal, models
from blackjax.mcmc.integrators import calls_per_integrator_step, generate_euclidean_integrator, generate_isokinetic_integrator, isokinetic_mclachlan, mclachlan_coefficients, name_integrator, omelyan_coefficients, velocity_verlet, velocity_verlet_coefficients, yoshida_coefficients
from blackjax.mcmc.mhmclmc import rescale
from blackjax.util import run_inference_algorithm



def get_num_latents(target):
  return target.ndims
#   return int(sum(map(np.prod, list(jax.tree_flatten(target.event_shape)[0]))))


def err(f_true, var_f, contract):
    """Computes the error b^2 = (f - f_true)^2 / var_f
        Args:
            f: E_sampler[f(x)], can be a vector
            f_true: E_true[f(x)]
            var_f: Var_true[f(x)]
            contract: how to combine a vector f in a single number, can be for example jnp.average or jnp.max
            
        Returns:
            contract(b^2)
    """    
    
    return jax.vmap(lambda f: contract(jnp.square(f - f_true) / var_f))



def grads_to_low_error(err_t, grad_evals_per_step= 1, low_error= 0.01):
    """Uses the error of the expectation values to compute the effective sample size neff
        b^2 = 1/neff"""
    
    cutoff_reached = err_t[-1] < low_error
    return find_crossing(err_t, low_error) * grad_evals_per_step, cutoff_reached
    
        
def calculate_ess(err_t, grad_evals_per_step, neff= 100):
    
    grads_to_low, cutoff_reached = grads_to_low_error(err_t, grad_evals_per_step, 1./neff)
    
    return (neff / grads_to_low) * cutoff_reached, grads_to_low*(1/cutoff_reached), cutoff_reached


def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    b = array > cutoff
    indices = jnp.argwhere(b)
    if indices.shape[0] == 0:
        print("\n\n\nNO CROSSING FOUND!!!\n\n\n", array, cutoff)
        return 1

    return jnp.max(indices)+1


def cumulative_avg(samples):
    return jnp.cumsum(samples, axis = 0) / jnp.arange(1, samples.shape[0] + 1)[:, None]


def gridsearch_tune(iterations, grid_size, model, sampler, batch, num_steps, center_L, center_step_size):
    results = defaultdict(float)
    iterations=2
    keys = jax.random.split(jax.random.PRNGKey(0), iterations+1)
    for i in range(iterations):
        # width = 2
        width = 2
        step_sizes = np.logspace(np.log10(center_step_size/width), np.log10(center_step_size*width), grid_size)
        Ls = np.logspace(np.log10(center_L/2), np.log10(center_L*2),grid_size)
        for j, (step_size, L) in enumerate(itertools.product(step_sizes , Ls)):
        
            ess, grad_calls_until_convergence = benchmark_chains(model, sampler(step_size=step_size, L=L), keys[i], n=num_steps, batch = batch) # batch=1000//model.ndims)
            results[(step_size, L)] = (ess, grad_calls_until_convergence)
            # print(j, grad_calls_until_convergence, step_size, L)

        
        best_ess, best_grads, (step_size, L) = max([(results[r][0], results[r][1], r) for r in results], key=operator.itemgetter(0))
        # raise Exception

        center_L, center_step_size = L, step_size

        # print(results)
        # print(f"best params on iteration {i} are stepsize {step_size} and L {L} with Grad Calls until Convergence {best_grads}")
        # print(f"L from ESS (0.4 * step_size/ESS): {0.4 * step_size/best_ess}")
    return center_L, center_step_size


def run_mhmclmc_no_tuning(coefficients, step_size, L):

    def s(logdensity_fn, num_steps, initial_position, transform, key):

        integrator = generate_isokinetic_integrator(coefficients)

        num_steps_per_traj = L/step_size
        alg = blackjax.mcmc.mhmclmc.mhmclmc(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(num_steps_per_traj)) ,
        integrator=integrator,
        )

        _, out, info = run_inference_algorithm(
        rng_key=key,
        initial_state_or_position=initial_position,
        inference_algorithm=alg,
        num_steps=num_steps, 
        transform=lambda x: transform(x.position), 
        progress_bar=True)

        return out, MCLMCAdaptationState(L=L, step_size=step_size, std_mat=1.), num_steps_per_traj * calls_per_integrator_step(coefficients)

    return s

def benchmark_chains(model, sampler, key, n=10000, batch=None, contract = jnp.average,):

    
    d = get_num_latents(model)
    if batch is None:
        batch = np.ceil(1000 / d).astype(int)
    key, init_key = jax.random.split(key, 2)
    keys = jax.random.split(key, batch)

    init_keys = jax.random.split(init_key, batch)
    init_pos = jax.vmap(model.sample_init)(init_keys)

    # samples, params, avg_num_steps_per_traj = jax.pmap(lambda pos, key: sampler(model.logdensity_fn, n, pos, model.transform, key))(init_pos, keys)
    samples, params, grad_calls_per_traj = jax.vmap(lambda pos, key: sampler(logdensity_fn=model.logdensity_fn, num_steps=n, initial_position= pos,transform= model.transform, key=key))(init_pos, keys)
    avg_grad_calls_per_traj = jnp.nanmean(grad_calls_per_traj, axis=0)
    # try:
    #     print(jnp.nanmean(params.step_size,axis=0), jnp.nanmean(params.L,axis=0))
    # except: pass
    
    full = lambda arr : err(model.E_x2, model.Var_x2, contract)(cumulative_avg(arr))
    err_t = jax.vmap(full)(samples**2)


    # outs = [calculate_ess(b, grad_evals_per_step=avg_grad_calls_per_traj) for b in err_t]
    # # print(outs[:10])
    # esses = [i[0].item() for i in outs if not math.isnan(i[0].item())]
    # grad_calls = [i[1].item() for i in outs if not math.isnan(i[1].item())]
    # return(mean(esses), mean(grad_calls))


    err_t_median = jnp.median(err_t, axis=0)
    esses, grad_calls, _ = calculate_ess(err_t_median, grad_evals_per_step=avg_grad_calls_per_traj)
    return esses, grad_calls




def run_benchmarks(batch_size):

    results = defaultdict(tuple)
    for variables in itertools.product(
        # ["mhmclmc", "nuts", "mclmc", ], 
        ["mhmclmc",], 
        # [StandardNormal(d) for d in np.ceil(np.logspace(np.log10(10), np.log10(10000), 10)).astype(int)],
        [MixedLogit()],
        # [velocity_verlet_coefficients, mclachlan_coefficients, yoshida_coefficients, omelyan_coefficients], 
        [mclachlan_coefficients], 
        ):

        sampler, model, coefficients = variables
        num_chains = 1 + batch_size//model.ndims

        print(f"\nModel: {model.name,model.ndims}, Sampler: {sampler}\n Coefficients: {coefficients}\nNumber of chains {num_chains}",) 

        key = jax.random.PRNGKey(2)
        for i in range(1):
            key1, key = jax.random.split(key)
            ess, grad_calls = benchmark_chains(model, partial(samplers[sampler], coefficients=coefficients),key1, n=500, batch=num_chains, contract=jnp.average)

            print(f"grads to low bias: {grad_calls}")

            results[((model.name, model.ndims), sampler, name_integrator(coefficients))] = (ess, grad_calls) 


            
    print(results)
            

    df = pd.Series(results).reset_index()
    df.columns = ["model", "sampler", "coeffs", "result"] 
    df.result = df.result.apply(lambda x: x[0].item())
    df.model = df.model.apply(lambda x: x[1])
    df.to_csv("results.csv", index=False)

    return results

def run_grid_search_benchmarks(batch_size):

    key = jax.random.PRNGKey(2)
    results = defaultdict(tuple)
    for variables in itertools.product(
        # ["mhmclmc", "nuts", "mclmc", ], 
        ["mhmclmc",], 
        [StandardNormal(d) for d in np.ceil(np.logspace(np.log10(10), np.log10(10000), 5)).astype(int)],
        # [velocity_verlet_coefficients, mclachlan_coefficients, yoshida_coefficients, omelyan_coefficients], 
        [mclachlan_coefficients, omelyan_coefficients], 
        ):

        sampler, model, coefficients = variables
        num_chains = 1 + batch_size//model.ndims

        current_key, key = jax.random.split(key) 
        init_pos_key, init_key, tune_key, bench_key = jax.random.split(current_key, 4)


        initial_position = model.sample_init(init_pos_key)

        initial_state = blackjax.mcmc.mhmclmc.init(
        position=initial_position, logdensity_fn=model.logdensity_fn, random_generator_arg=init_key
        )

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size: blackjax.mcmc.mhmclmc.build_kernel(
                    integrator=generate_isokinetic_integrator(coefficients),
                    integration_steps_fn = lambda k : jnp.ceil(jax.random.uniform(k) * rescale(avg_num_integration_steps))
                )(
                    rng_key=rng_key, 
                    state=state, 
                    step_size=step_size, 
                    logdensity_fn=model.logdensity_fn)

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.adaptation.mclmc_adaptation.mhmclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=1000,
            state=initial_state,
            rng_key=tune_key,
            target=target_acceptance_rate_of_order[integrator_order(coefficients)],
            frac_tune1=0.1,
            frac_tune2=0.1,
            frac_tune3=0.0,
        )

        print(f"\nModel: {model.name,model.ndims}, Sampler: {sampler}\n Coefficients: {coefficients}\nNumber of chains {num_chains}",) 
        print(f"params after initial tuning are L={blackjax_mclmc_sampler_params.L}, step_size={blackjax_mclmc_sampler_params.step_size}")


        L, step_size = gridsearch_tune(iterations=2, grid_size=5, model=model, sampler=partial(run_mhmclmc_no_tuning, coefficients=coefficients), batch=num_chains, num_steps=1000, center_L=blackjax_mclmc_sampler_params.L, center_step_size=blackjax_mclmc_sampler_params.step_size)
        print(f"params after grid tuning are L={L}, step_size={step_size}")


        bench_key 
        for i in range(1):
            key1, bench_key = jax.random.split(bench_key)
            ess, grad_calls = benchmark_chains(model, run_mhmclmc_no_tuning(coefficients=coefficients, L=L, step_size=step_size),key1, n=1000, batch=num_chains, contract=jnp.average)

            print(f"grads to low bias: {grad_calls}")

            results[((model.name, model.ndims), sampler, name_integrator(coefficients))] = (ess, grad_calls) 


            
    print(results)
            

    df = pd.Series(results).reset_index()
    df.columns = ["model", "sampler", "coeffs", "result"] 
    df.result = df.result.apply(lambda x: x[0].item())
    df.model = df.model.apply(lambda x: x[1])
    df.to_csv("results.csv", index=False)

    return results

if __name__ == "__main__":

    # run_grid_search_benchmarks(batch_size=1000)
    run_benchmarks(batch_size=10000)


