from functools import partial
from blackjax.mcmc import hmc
import jax
# 
import jax.numpy as jnp
import blackjax.mcmc.nuts as nuts
import blackjax

# def logdensity_fn(x):
#         mu2 = 0.03 * (x[0] ** 2 - 100)
#         return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))


# def logdensity_fn_joint(x,y):
#         mu2 = 0.03 * (x[0] ** 2 - 100)
#         return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(y[0] - mu2))
    
def logdensity_fn_joint(pos):
        # jax.debug.print("pos {x}", x=pos)
        mu2 = 0.03 * (pos['x'] ** 2 - 100)
        return -0.5 * (jnp.square(pos['x'] / 10.0) + jnp.square(pos['y'] - mu2))


# pos = {'x': 1.0}

# print(state)

# state = hmc.init({'x': 1.0, 'y': 1.0}, logdensity_fn=lambda pos: logdensity_fn_joint({'x': pos['x'], 'y': 0.}))
# pos = {'x': 1.0, 'y': 1.0}

# lfunc = lambda pos: logdensity_fn_joint({'x': pos['x'], 'y': state.position['y']})

# print(lfunc(state.position))

# print(jax.grad(lfunc)(state.position))

# curried_logdensity_fn = lambda state: lambda pos: logdensity_fn_joint({'x': pos['x'], 'y': state.position['y']})

# print(state)



@jax.jit
def gibbs_kernel(rng_key, pos_joint):
  
    rng_key_1, rng_key_2 = jax.random.split(rng_key)

    x_state = hmc.init(pos_joint['x'], logdensity_fn=lambda x_pos : logdensity_fn_joint({'x': x_pos, 'y': pos_joint['y']}) )

    # print(x_state)

    new_state, info = blackjax.nuts.build_kernel()(
        rng_key=rng_key_1,
        state=x_state,
        logdensity_fn=lambda x_pos : logdensity_fn_joint({'x': x_pos, 'y': pos_joint['y']}),
        
        # logdensity_fn_joint(hmc.HMCState(position={'x': state.position['x'], 'y': pos['y']}, logdensity=state.logdensity, logdensity_grad={'x': state.logdensity_grad['x'], 'y': state.logdensity_grad['y']})),
        step_size=0.1,
        # num_integration_steps=1,
        inverse_mass_matrix=jnp.array([1.0]),
        # inverse_mass_matrix=jax.tree.map(lambda x: 1.0, state.position),
    )


    pos_joint['x'] = new_state.position

    y_state = hmc.init(pos_joint['y'], logdensity_fn=lambda y_pos : logdensity_fn_joint({'x': pos_joint['x'], 'y': y_pos}) ) 
    
    new_state, info = blackjax.nuts.build_kernel()(
        rng_key=rng_key_2,
        state=y_state,
        logdensity_fn=lambda y_pos : logdensity_fn_joint({'x': pos_joint['x'], 'y': y_pos}),
        
        # logdensity_fn_joint(hmc.HMCState(position={'x': state.position['x'], 'y': pos['y']}, logdensity=state.logdensity, logdensity_grad={'x': state.logdensity_grad['x'], 'y': state.logdensity_grad['y']})),
        step_size=0.1,
        # num_integration_steps=1,
        inverse_mass_matrix=jnp.array([1.0]),
        # inverse_mass_matrix=jax.tree.map(lambda x: 1.0, state.position),
    )

    pos_joint['y'] = new_state.position

    return pos_joint

print(gibbs_kernel(jax.random.key(0), {'x': 1.0, 'y': 1.0}))

def posterior_draw(key):
        z = jax.random.normal(key, shape = (2, ))
        x0 = 10.0 * z[0]
        x1 = 0.03 * (x0 ** 2 - 100) + z[1]
        return jnp.array([x0, x1])

results = {'x': [], 'y': []}
pos_joint_arr = posterior_draw(jax.random.key(0))
pos_joint = {'x': pos_joint_arr[0], 'y': pos_joint_arr[1]}
# pos_joint = {'x': 0.0, 'y': 0.0}
for i in range(5000):
        print(pos_joint)
        pos_joint = gibbs_kernel(jax.random.key(i+1), pos_joint)
        results['x'].append(pos_joint['x'].item())
        results['y'].append(pos_joint['y'].item())


import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x=results['x'], y=results['y'])
# save plot
plt.savefig('gibbs_hmc.png')

#     # new_full_state, info = alg.step(rng_key, hmc.HMCState(
#     #        position=jnp.concat([state_x.position, state_y.position], axis=0), 
#     #        logdensity=state_x.logdensity+state_y.logdensity, 
#     #        logdensity_grad=jnp.concat([state_x.logdensity_grad, state_y.logdensity_grad], axis=0),
#     # ))
    

#     print(new_full_state.position.shape, "bar")

#     new_state_x = hmc.HMCState(
#         position=new_full_state.position[:1],
#         logdensity=new_full_state.logdensity,
#         logdensity_grad=new_full_state.logdensity_grad[:1],
#     ) 

#     return (new_state_x, new_state_y), info 
    
  

# (next_state_x, next_state_y), info = gibbs_kernel(rng_key, (state_x, state_y))


# import numpy as np
# num_samples = 100
# x_poss = np.zeros((num_samples,))
# y_poss = np.zeros((num_samples,))
# # run this kernel for n steps
# for i in range(num_samples):
#     (next_state_x, next_state_y), info = gibbs_kernel(rng_key, (next_state_x, next_state_y))
#     # x_poss.at[i].set(next_state_x.position[0])
#     # y_poss.at[i].set(next_state_y.position[0])

#     x_poss[i] = next_state_x.position[0].item()
#     y_poss[i] = next_state_y.position[0].item()

# # # next_state, info = alg.step(rng_key, initial_state)

# # scatterplot 
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.scatterplot(x=x_poss, y=y_poss)
# # save plot
# plt.savefig('gibbs_hmc.png')

# # onsager idea brainstorm: 
