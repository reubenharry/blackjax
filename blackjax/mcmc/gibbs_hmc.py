from functools import partial
from blackjax.mcmc import hmc
import jax
# 
import jax.numpy as jnp
import blackjax.mcmc.nuts as nuts
import blackjax

def logdensity_fn(x):
        mu2 = 0.03 * (x[0] ** 2 - 100)
        return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))


def logdensity_fn_joint(x,y):
        mu2 = 0.03 * (x[0] ** 2 - 100)
        return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(y[0] - mu2))
    

def logdensity_fn_x(y):
        return lambda x : -0.5 * (jnp.square(x / 10.0) + jnp.square(y - (0.03 * (x ** 2 - 100))))

def logdensity_fn_x_easy(x):
        mu2 = 0.03 * (x[0] ** 2 - 100)
        return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(1.0 - mu2))
    
# Define the initial state
rng_key = jax.random.PRNGKey(0)
initial_position = jnp.array([1.0, 1.0])

alg = blackjax.nuts(
            logdensity_fn=logdensity_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.array([1.0, 1.0]),
        )

initial_state = alg.init(initial_position, rng_key)


kernel = blackjax.nuts.build_kernel()

position_x = jnp.array([0.1])
position_y = jnp.array([0.2])


# print(logdensity_fn_x_easy(jnp.array([1.0])))
flip = lambda f: lambda x, y: f(x=y, y=x)

state_x = hmc.init(position_x, logdensity_fn=partial(logdensity_fn_joint, y=jnp.array([1])))
state_y = hmc.init(position_y, logdensity_fn=partial(flip(logdensity_fn_joint), y=jnp.array([2])))


# state_x = hmc.init(position_x, logdensity_fn=logdensity_fn_x_easy)
# state_x = hmc.init(position_x, logdensity_fn=logdensity_fn_x_easy)

# print(state_x)
@jax.jit
def gibbs_kernel(rng_key, state):
    state_x, state_y = state
    print(state_x.position.shape)
    rng_key_1, rng_key_2 = jax.random.split(rng_key)
    # print(state_x.position[1])
    # new_state_x, info = kernel(
    #        rng_key=rng_key_1,
    #        state=state_x,
    #        logdensity_fn=partial(logdensity_fn_joint, y=state_y.position),
    #     #    logdensity_fn=logdensity_fn_x(y=position[1]),
    #        step_size=0.1,
    #        inverse_mass_matrix=jnp.array([1.0]),

    # )

    # new_state_y, info = kernel(
    #        rng_key=rng_key_2,
    #        state=state_y,
    #        logdensity_fn=partial(flip(logdensity_fn_joint),y=new_state_x.position),
    #     #    logdensity_fn=logdensity_fn_x(y=position[1]),
    #        step_size=0.1,
    #        inverse_mass_matrix=jnp.array([1.0]),

    # )


    new_full_state, info = alg.step(rng_key, hmc.HMCState(
           position=jnp.concat([state_x.position, state_y.position], axis=0), 
           logdensity=state_x.logdensity+state_y.logdensity, 
           logdensity_grad=jnp.concat([state_x.logdensity_grad, state_y.logdensity_grad], axis=0),
    ))
    

    print(new_full_state.position.shape, "bar")

    new_state_x = hmc.HMCState(
        position=new_full_state.position[:1],
        logdensity=new_full_state.logdensity,
        logdensity_grad=new_full_state.logdensity_grad[:1],
    ) 

    return (new_state_x, new_state_y), info 
    
  

(next_state_x, next_state_y), info = gibbs_kernel(rng_key, (state_x, state_y))


import numpy as np
num_samples = 100
x_poss = np.zeros((num_samples,))
y_poss = np.zeros((num_samples,))
# run this kernel for n steps
for i in range(num_samples):
    (next_state_x, next_state_y), info = gibbs_kernel(rng_key, (next_state_x, next_state_y))
    # x_poss.at[i].set(next_state_x.position[0])
    # y_poss.at[i].set(next_state_y.position[0])

    x_poss[i] = next_state_x.position[0].item()
    y_poss[i] = next_state_y.position[0].item()

# # next_state, info = alg.step(rng_key, initial_state)

# scatterplot 
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x=x_poss, y=y_poss)
# save plot
plt.savefig('gibbs_hmc.png')

# onsager idea brainstorm: 
