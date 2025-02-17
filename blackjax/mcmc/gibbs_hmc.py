from functools import partial

from blackjax.util import run_inference_algorithm
import jax

#
import jax.numpy as jnp

import blackjax
import blackjax.mcmc.nuts as nuts
from blackjax.mcmc import hmc

# def logdensity_fn(x):
#         mu2 = 0.03 * (x[0] ** 2 - 100)
#         return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))


# def logdensity_fn_joint(x,y):
#         mu2 = 0.03 * (x[0] ** 2 - 100)
#         return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(y[0] - mu2))


def logdensity_fn_joint(pos):
    # jax.debug.print("pos {x}", x=pos)
    mu2 = 0.03 * (pos["x"] ** 2 - 100)
    return -0.5 * (jnp.square(pos["x"] / 10.0) + jnp.square(pos["y"] - mu2))


# pos = {'x': 1.0}

# print(state)

# state = hmc.init({'x': 1.0, 'y': 1.0}, logdensity_fn=lambda pos: logdensity_fn_joint({'x': pos['x'], 'y': 0.}))
# pos = {'x': 1.0, 'y': 1.0}

# lfunc = lambda pos: logdensity_fn_joint({'x': pos['x'], 'y': state.position['y']})

# print(lfunc(state.position))

# print(jax.grad(lfunc)(state.position))

# curried_logdensity_fn = lambda state: lambda pos: logdensity_fn_joint({'x': pos['x'], 'y': state.position['y']})

# print(state)

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
    


def posterior_draw(key):
    z = jax.random.normal(key, shape=(2,))
    x0 = 10.0 * z[0]
    x1 = 0.03 * (x0**2 - 100) + z[1]
    return jnp.array([x0, x1])


results = {"x": [], "y": []}
pos_joint_arr = posterior_draw(jax.random.key(0))
pos_joint = {"x": pos_joint_arr[0], "y": pos_joint_arr[1]}
joint_state = hmc.init(pos_joint, logdensity_fn=logdensity_fn_joint)

print(joint_state)

# pos_joint = {'x': 0.0, 'y': 0.0}
for i in range(500):
    joint_state = gibbs_kernel(jax.random.key(i + 1), joint_state)
    results["x"].append(joint_state.position["x"].item())
    results["y"].append(joint_state.position["y"].item())


import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=results["x"], y=results["y"])
# save plot
plt.savefig("gibbs_hmc.png")
