
from blackjax.adaptation.ensemble_mclmc import Adaptation
import jax
import jax.numpy as jnp
from jax import device_put, lax, vmap
from jax.experimental.shard_map import shard_map
from jax.random import split
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_map

import blackjax.adaptation.ensemble_umclmc as umclmc
from blackjax.mcmc.adjusted_mclmc import build_kernel as build_kernel_malt
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import (
    generate_isokinetic_integrator,
    mclachlan_coefficients,
)


def eca_step(
    kernel, summary_statistics_fn, adaptation_update, num_chains, ensemble_info=None
):
    """
    Construct a single step of ensemble chain adaptation (eca) to be performed in parallel on multiple devices.
    """

    def _step(state_all, xs):
        """This function operates on a single device."""
        (
            state,
            adaptation_state,
        ) = state_all  # state is an array of states, one for each chain on this device. adaptation_state is the same for all chains, so it is not an array.
        (
            _,
            keys_sampling,
            key_adaptation,
        ) = xs  # keys_sampling.shape = (chains_per_device, )

        # update the state of all chains on this device
        state, info = vmap(kernel, (0, 0, None))(keys_sampling, state, adaptation_state)


        # combine all the chains to compute expectation values
        theta = vmap(summary_statistics_fn, (0, 0, None))(state, info, key_adaptation)
        Etheta = tree_map(
            lambda theta: lax.psum(jnp.sum(theta, axis=0), axis_name="chains")
            / num_chains,
            theta,
        )

        adaptation_state, info_to_be_stored = adaptation_update(
            adaptation_state, Etheta
        )

        return (state, adaptation_state), info_to_be_stored

    if ensemble_info is not None:

        def step(state_all, xs):
            (state, adaptation_state), info_to_be_stored = _step(state_all, xs)
            return (state, adaptation_state), (
                info_to_be_stored,
                vmap(ensemble_info)(state.position),
            )

        return step

    else:
        return _step



def run_eca(
    rng_key,
    initial_state,
    kernel,
    adaptation,
    num_steps,
    num_chains,
    mesh,
    ensemble_info=None,
    early_stop=False,
):

    step = eca_step(
        kernel,
        adaptation.summary_statistics_fn,
        adaptation.update,
        num_chains,
        ensemble_info,
    )

    def all_steps(initial_state, keys_sampling, keys_adaptation):
        """This function operates on a single device. key is a random key for this device."""

        initial_state_all = (initial_state, adaptation.initial_state)
        # return (initial_state, adaptation.initial_state, None)

        # run sampling
        xs = (
            jnp.arange(num_steps),
            keys_sampling.T,
            keys_adaptation,
        )  # keys for all steps that will be performed. keys_sampling.shape = (num_steps, chains_per_device), keys_adaptation.shape = (num_steps, )

        # ((a, Int) -> (a, Int))
        def step_while(a):
            x, i, _ = a

            auxilliary_input = (xs[0][i], xs[1][i], xs[2][i])

            output, info = step(x, auxilliary_input)

            return (output, i + 1, info[0].get("while_cond"))

        if early_stop:
            final_state_all, i, _ = lax.while_loop(
                lambda a: ((a[1] < num_steps) & a[2]),
                step_while,
                (initial_state_all, 0, True),
            )
            info_history = None

        else:
            final_state_all, info_history = lax.scan(step, initial_state_all, xs)

        final_state, final_adaptation_state = final_state_all
        return (
            final_state,
            final_adaptation_state,
            info_history,
        )  # info history is composed of averages over all chains, so it is a couple of scalars

    p, pscalar = PartitionSpec("chains"), PartitionSpec()
    parallel_execute = shard_map(
        all_steps,
        mesh=mesh,
        in_specs=(p, p, pscalar),
        out_specs=(p, pscalar, pscalar),
        check_rep=False,
    )

    # produce all random keys that will be needed

    key_sampling, key_adaptation = split(rng_key)
    num_steps = jnp.array(num_steps).item()
    keys_adaptation = split(key_adaptation, num_steps)
    distribute_keys = lambda key, shape: device_put(
        split(key, shape), NamedSharding(mesh, p)
    )  # random keys, distributed across devices
    keys_sampling = distribute_keys(key_sampling, (num_chains, num_steps))

    # run sampling in parallel
    final_state, final_adaptation_state, info_history = parallel_execute(
        initial_state, keys_sampling, keys_adaptation
    )

    return final_state, final_adaptation_state, info_history



build_kernel = lambda logdensity_fn, integrator, inverse_mass_matrix: lambda key, state, adap: build_kernel_malt(
    logdensity_fn=logdensity_fn,
    integrator=integrator,
    inverse_mass_matrix=inverse_mass_matrix,
)(
    rng_key=key,
    state=state,
    step_size=adap.step_size,
    num_integration_steps=adap.steps_per_sample,
    L_proposal_factor=1.25,
)


def logdensity_fn(x):
    mu2 = 0.03 * (x[0] ** 2 - 100)
    return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))


def transform(x):
    return x


def sample_init(key):
    z = jax.random.normal(key, shape=(2,))
    x0 = 10.0 * z[0]
    x1 = 0.03 * (x0**2 - 100) + z[1]
    return jnp.array([x0, x1])


num_chains = 128

mesh = jax.sharding.Mesh(devices=jax.devices(), axis_names="chains")

key_init, key_umclmc, key_mclmc = jax.random.split(jax.random.key(0), 3)

integrator_coefficients = mclachlan_coefficients

# initialize the chains
initial_state = umclmc.initialize(
    key_init, logdensity_fn, sample_init, num_chains, mesh
)

diagonal_preconditioning = False
ndims = 2

alpha = 1.9
C = 0.1
r_end = 5e-3
ensemble_observables = lambda x: x

# burn-in with the unadjusted method #
kernel = umclmc.build_kernel(logdensity_fn)
save_num = 20  # (int)(jnp.rint(save_frac * num_steps1))
adap = umclmc.Adaptation(
    ndims,
    alpha=alpha,
    bias_type=3,
    save_num=save_num,
    C=C,
    power=3.0 / 8.0,
    r_end=r_end,
    observables_for_bias=lambda position: jnp.square(
        transform(jax.flatten_util.ravel_pytree(position)[0])
    ),
)



### I think the problem is using `final_state` in the subsequent computation
final_state, final_adaptation_state, info1 = run_eca(
    key_umclmc,
    initial_state,
    kernel,
    adap,
    100,
    num_chains,
    mesh,
    ensemble_observables,
    early_stop=True,
)



def all_steps(initial_state, keys_sampling, keys_adaptation):
    """This function operates on a single device. key is a random key for this device."""
    
    # refine the results with the adjusted method
    _acc_prob = 0.9

    integrator = generate_isokinetic_integrator(mclachlan_coefficients)
    gradient_calls_per_step = (
        len(mclachlan_coefficients) // 2
    )  # scheme = BABAB..AB scheme has len(scheme)//2 + 1 Bs. The last doesn't count because that gradient can be reused in the next step.

    inverse_mass_matrix = 1.0


    steps_per_sample = 15
    num_steps2 = 100

    num_samples = num_steps2 // (gradient_calls_per_step * steps_per_sample)
    num_adaptation_samples = (
        num_samples // 2
    )  # number of samples after which the stepsize is fixed.


    initial_state = HMCState(
    final_state.position, final_state.logdensity, final_state.logdensity_grad
    )

    adaptation= Adaptation(
    final_adaptation_state,
    num_adaptation_samples,
    steps_per_sample,
    _acc_prob,
    )
    ensemble_info = None
    step = eca_step(
        build_kernel(
        logdensity_fn, integrator, inverse_mass_matrix=inverse_mass_matrix
        ),
        adaptation.summary_statistics_fn,
        adaptation.update,
        num_chains,
        ensemble_info,
    )

    initial_state_all = (initial_state, adaptation.initial_state)
    # return (initial_state, adaptation.initial_state, None)

    # run sampling
    xs = (
        jnp.arange(num_steps),
        keys_sampling.T,
        keys_adaptation,
    )  # keys for all steps that will be performed. keys_sampling.shape = (num_steps, chains_per_device), keys_adaptation.shape = (num_steps, )

    # ((a, Int) -> (a, Int))
    def step_while(a):
        x, i, _ = a

        auxilliary_input = (xs[0][i], xs[1][i], xs[2][i])

        output, info = step(x, auxilliary_input)

        return (output, i + 1, info[0].get("while_cond"))

    final_state_all, info_history = lax.scan(step, initial_state_all, xs)

    final_state_new, final_adaptation_state_new = final_state_all
    return (
        final_state_new,
        final_adaptation_state_new,
        info_history,
    )  # info history is composed of averages over all chains, so it is a couple of scalars

p, pscalar = PartitionSpec("chains"), PartitionSpec()
parallel_execute = shard_map(
    all_steps,
    mesh=mesh,
    in_specs=(p, p, pscalar),
    out_specs=(p, pscalar, pscalar),
    check_rep=False,
)

# produce all random keys that will be needed

key_sampling, key_adaptation = split(jax.random.key(0))
num_steps = jnp.array(200).item()
keys_adaptation = split(key_adaptation, num_steps)
distribute_keys = lambda key, shape: device_put(
    split(key, shape), NamedSharding(mesh, p)
)  # random keys, distributed across devices
keys_sampling = distribute_keys(key_sampling, (num_chains, num_steps))

# run sampling in parallel
final_state, final_adaptation_state, info_history = parallel_execute(
    initial_state, keys_sampling, keys_adaptation
)



##########################









####
# This one fails
####

# final_state, final_adaptation_state, info2 = run_eca(
#     rng_key=key_mclmc,
#     initial_state= HMCState(
#     final_state.position, final_state.logdensity, final_state.logdensity_grad
#     ),
#     kernel = build_kernel(
#     logdensity_fn, integrator, inverse_mass_matrix=inverse_mass_matrix
#     ),
#     adaptation= Adaptation(
#     final_adaptation_state,
#     num_adaptation_samples,
#     steps_per_sample,
#     _acc_prob,
#     ),
#     num_steps=num_samples,
#     num_chains=num_chains,
#     mesh=mesh,
#     ensemble_info=ensemble_observables,
# )
