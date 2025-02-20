from functools import partial

from blackjax.util import run_inference_algorithm
import jax

#
import jax.numpy as jnp

import blackjax
import blackjax.mcmc.nuts as nuts
from blackjax.mcmc import hmc, mala

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def badBanana(pos): 
    return - 0.5 * ( (pos[0] / 10.0)**2 + pos[1]**2  )

def bananaCorrection(pos):  
    curvature = 0.03
    return - 0.5 * ((curvature * (pos[0] ** 2 - 100))**2 - 2* pos[1]* (curvature * (pos[0]**2 - 100)))



def goodBanana(pos):
    mu2 = 0.03 * (pos[0] ** 2 - 100)
    return -0.5 * (jnp.square(pos[0] / 10.0) + jnp.square(pos[1] - mu2))

# def posterior_draw(key):
#     z = jax.random.normal(key, shape=(2,))
#     x0 = 10.0 * z[0]
#     x1 = 0.03 * (x0**2 - 100) + z[1]
#     return jnp.array([x0, x1])


keys = jax.random.split(jax.random.key(0), 100)
pos = jax.vmap(lambda k: jax.random.normal(key=k, shape=(2,)))(keys)

# print(hmc.init(position=pos, logdensity_fn=logdensity_fn_joint))
# pos_joint = {"x": pos_joint_arr[0], "y": pos_joint_arr[1]}
state = jax.vmap(lambda p: mala.init(position=p, logdensity_fn=badBanana))(pos)
# print(state)

step_size = 0.05

onsager_function = lambda xd, grad_corr, grad_b : step_size * (-0.25) * (grad_corr @ grad_corr  - 2 * xd @ grad_corr + 2 * grad_b @ grad_corr)

@jax.jit
def onsager_update(rng_key, states):

    alg = blackjax.mala(
        # logdensity_fn=goodBanana,
        logdensity_fn=badBanana,
        step_size=step_size,
        unadjusted=True,
    )

    sample_key, rng_key = jax.random.split(rng_key)

    keys = jax.random.split(rng_key, 100)

    

    new_state, (new_states, info) = jax.vmap(lambda st, k : run_inference_algorithm(
        rng_key=k,
        inference_algorithm=alg,
        initial_state=st,
        num_steps=1,
    ))(states, keys)

    print(new_states.position.shape, "new states shape")

    xdot = ((new_state.position - states.position) / step_size)


    grad_correction = (jax.vmap(jax.grad(bananaCorrection)))(new_state.position)
    grad_bad = (jax.vmap(jax.grad(badBanana)))(new_state.position)

    print(xdot.shape, "xdot shape", grad_correction.shape, "grad correction shape", grad_bad.shape, "grad bad shape")
    logits = jax.vmap(onsager_function)(xdot, grad_correction, grad_bad)

    print(logits.shape, "logits shape")

    # logits = logits.sum(axis=1)

    print(logits.shape,"bar")
    indices = tfd.Categorical(
        logits=logits,
        ).sample(logits.shape[0], seed=sample_key)

    jax.debug.print("indices {x}", x=indices)

    batch = False
    if batch:

        xdot = ((new_states.position[:, 1:, :] - new_states.position[:, :-1, :]) / step_size)

        grad_correction = jax.vmap(jax.vmap(jax.grad(bananaCorrection)))(new_states.position[:,:-1, :])
        grad_bad = jax.vmap(jax.vmap(jax.grad(badBanana)))(new_states.position[:,:-1, :])

        logits = jax.vmap(jax.vmap(onsager_function))(xdot, grad_correction, grad_bad)

        logits = logits.sum(axis=1)

        print(logits.shape,"bar")
        indices = tfd.Categorical(
            logits=logits,
            ).sample(logits.shape[0], seed=sample_key)

        jax.debug.print("indices {x}", x=indices)
    

    # print(xdot.shape, grad_correction.shape, grad_bad.shape)

    new_position = new_state.position[indices]
    # new_position = new_state.position

    # print(new_position.shape)

    new_states = mala.MALAState(new_position, new_state.logdensity, new_state.logdensity_grad)

    return new_states

new_states = state

full_new_states = jnp.zeros((1000, 100, 2))
for i in range(1000):
    new_states = onsager_update(jax.random.key(i), new_states)
    full_new_states = full_new_states.at[i, :, :].set(new_states.position)
# new_states = state
# plot states with scatterplot

import matplotlib.pyplot as plt
import seaborn as sns

full_new_states = full_new_states.reshape((100000, 2))

sns.scatterplot(x=full_new_states[:, 0], y=full_new_states[:, 1])
# sns.scatterplot(x=new_states.position[:, 0], y=new_states.position[:, 1])
# save plot
plt.savefig("onsager.png")

# this is wrong


# print(grad_correction.shape, grad_bad.shape, xdot.shape)

# 




raise Exception

def partial_update(rng_key, joint_state, fixed_var, changing_var, num_steps=1, step_size=0.05):

    partial_state = hmc.HMCState(
        position=joint_state.position[changing_var],
        logdensity=joint_state.logdensity,
        logdensity_grad=joint_state.logdensity_grad[changing_var],
    )

    alg = blackjax.nuts(
        logdensity_fn=lambda pos: logdensity_fn_joint(
            {changing_var: pos, fixed_var: joint_state.position[fixed_var]}
        ),
        step_size=0.05,
        inverse_mass_matrix=jnp.array([1.0]),
    )

    new_partial_state = run_inference_algorithm(
        rng_key=rng_key,
        inference_algorithm=alg,
        initial_state=partial_state,
        num_steps=10,
    )[0]

    joint_state = hmc.HMCState(
        position={changing_var: new_partial_state.position, fixed_var: joint_state.position[fixed_var]},
        logdensity=new_partial_state.logdensity,
        logdensity_grad={changing_var: new_partial_state.logdensity_grad, fixed_var: joint_state.logdensity_grad[fixed_var]},
    )

    return joint_state


@jax.jit
def gibbs_kernel(rng_key, joint_state):

    rng_key_1, rng_key_2 = jax.random.split(rng_key)

    joint_state = partial_update(rng_key_1, joint_state, "y", "x", num_steps=1)
    joint_state = partial_update(rng_key_2, joint_state, "x", "y", num_steps=10)
    return joint_state
    




# results = {"x": [], "y": []}

# print(joint_state)

# # pos_joint = {'x': 0.0, 'y': 0.0}
# for i in range(500):
#     joint_state = gibbs_kernel(jax.random.key(i + 1), joint_state)
#     results["x"].append(joint_state.position["x"].item())
#     results["y"].append(joint_state.position["y"].item())


# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.scatterplot(x=results["x"], y=results["y"])
# # save plot
# plt.savefig("gibbs_hmc.png")


## 0 split the banana
## 1. sample n chains
## 2. take differences
## 3. take gradients
## 4. calculate the onsager weight
## 5. multinomial resampling
## 6. wrap this in a kernel
## 7. plot results
