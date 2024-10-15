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
#"""Public API for the MCLMC Kernel"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from typing import Callable, NamedTuple, Any

from blackjax.mcmc.integrators import IntegratorState, isokinetic_velocity_verlet
from blackjax.types import Array, ArrayLike
from blackjax.util import pytree_size

from blackjax.mcmc import mclmc

from blackjax.mcmc.integrators import _normalized_flatten_array




def no_nans(a):
    flat_a, unravel_fn = ravel_pytree(a)
    return jnp.all(jnp.isfinite(flat_a))


def nan_reject(nonans, old, new):
    """Equivalent to
        return new if nonans else old"""
        
    return jax.lax.cond(nonans, lambda _: new, lambda _: old, operand=None)



def build_kernel(logdensity_fn):
    """MCLMC kernel (with nan rejection)"""
    
    kernel = mclmc.build_kernel(logdensity_fn= logdensity_fn, integrator= isokinetic_velocity_verlet)
    
    
    def sequential_kernel(key, state, hyp):
        
        new_state, info = kernel(key, state, hyp.L, hyp.step_size)
        
        # reject the new state if there were nans
        nonans = no_nans(new_state)
        new_state = nan_reject(nonans, state, new_state)
        
        return new_state, {'nans': 1-nonans, 'energy_change': info.energy_change * nonans}

    
    return sequential_kernel



    
class Initialization:
    """initialize the chains based on the equipartition of the initial condition.
       We initialize the velocity along grad log p if E_ii > 1 and along -grad log p if E_ii < 1.
    """
    
    def __init__(self, logdensity_fn, sample_init):
        self.logdensity_fn = logdensity_fn
        self.sample_init = sample_init
        
        
    def sequential_init(self, key):
        """initialize the position using sample_init and the velocity along the gradient"""
        position = self.sample_init(key)
        logdensity, logdensity_grad = jax.value_and_grad(self.logdensity_fn)(position)
        flat_g, unravel_fn = ravel_pytree(logdensity_grad)
        velocity = unravel_fn(_normalized_flatten_array(flat_g)[0]) # = grad logp/ |grad logp|
        return IntegratorState(position, velocity, logdensity, logdensity_grad)
    
    def summary_statistics(self, state):
        """compute the diagonal elements of the equipartition matrix"""
        return -state.position * state.logdensity_grad
    
    def ensemble_init(self, state, equipartition):
        """flip the velocity, depending on the equipartition condition"""
        signs = -2. * (equipartition < 1.) + 1.    
        velocity = jax.tree_util.tree_map(lambda sign, u: sign * u, signs, state.momentum)
        return IntegratorState(state.position, velocity, state.logdensity, state.logdensity_grad)



class Hyperparameters(NamedTuple):
    
    L: float
    step_size: float
    

class AdaptationState(NamedTuple):
    steps: int
    EEVPD: float
    EEVPD_wanted: float
    #history: Array
    
    hyperparameters: Any
    

def equipartition_diagonal(state):
    """Ei = E_ensemble (- grad log p_i x_i ). Ei is 1 if we have converged. 
    virial_loss = average over parameters (Ei)"""
    return jax.tree_util.tree_map(lambda x, g: -jnp.average(x * g, axis= 0), state.position, state.logdensity_grad)

def equipartition_diagonal_loss(Eii):
    Eii_flat, unravel_fn = ravel_pytree(Eii)
    return jnp.average(jnp.square(1.- Eii_flat))



class Adaptation:
    
    def __init__(self, num_dims, 
                 alpha= 1., C= 0.1, 
                 monitor_exp_vals= lambda x: 0., contract_exp_vals= lambda exp_vals: 0.
                 ):

        self.num_dims = num_dims
        self.alpha = alpha
        self.C = C
        self.monitor_exp_vals = monitor_exp_vals
        self.contract_exp_vals = contract_exp_vals

        #delay_num = (int)(jnp.rint(delay_frac * num_steps))    
        #sigma = unravel_fn(jnp.ones(flat_pytree.shape, dtype = flat_pytree.dtype))
        
        hyp = Hyperparameters(L= jnp.inf, step_size= 0.01 * jnp.sqrt(num_dims)) # do not add noise for the first step
        
        #history = jnp.inf * jnp.ones(delay_num)
        #history = jnp.concatenate((jnp.ones(1) * 1e50, jnp.ones(delay_num-1) * jnp.inf)) # loss history
        
        self.initial_state = AdaptationState(0, 1e-3, 1e-3, hyp)
        
        
    def summary_statistics(self, state, info):
    
        position_flat, unravel_fn = ravel_pytree(state.position)
        
        return {'equipartition_diagonal': equipartition_diagonal(state), 
                'x': position_flat, 'xsq': jnp.square(position_flat), 
                'E': info['energy_change'], 'Esq': jnp.square(info['energy_change']),
                'rejection_rate_nans': info['nans'],
                'monitored_exp_vals': self.monitor_exp_vals(state.position)
                }
        
        
    def update(self, adaptation_state, Etheta, key_adaptation):

        hyp = adaptation_state.hyperparameters
        
        # combine the expectation values to get useful scalars
        equi_diag = equipartition_diagonal_loss(Etheta['equipartition_diagonal'])
        L = self.alpha * jnp.sqrt(jnp.sum(Etheta['xsq'] - jnp.square(Etheta['x']))) # average over the ensemble, sum over parameters (to get sqrt(d))
        EEVPD = (Etheta['Esq'] - jnp.square(Etheta['E'])) / self.num_dims
        nans = Etheta['rejection_rate_nans'] > 0
        contracted_exp_vals = self.contract_exp_vals(Etheta['monitored_exp_vals'])
        

        # hyperparameter adaptation                                              
        bias = equi_diag #estimate the bias from the equipartition loss
        EEVPD_wanted = self.C * jnp.power(bias, 3./8.)
        
        eps_factor = jnp.power(EEVPD_wanted / EEVPD, 1./6.)
        eps_factor = nans * 0.5 + (1-nans) * eps_factor # reduce the stepsize if there were nans
        eps_factor = jnp.clip(eps_factor, 0.3, 3.)
        
        hyp = Hyperparameters(L, hyp.step_size * eps_factor)
        
        # determine if we want to finish this stage (= if loss is no longer decreassing)
        #history = jnp.concatenate((jnp.ones(1) * bias, adap_state.history[:-1]))
        #decreasing = (history[-1] > history[0]) or (adap_state.steps < adap_state.history.shape[0])
        #cond = decreasing and (adap_state.steps < max_iter)
        #cond = (adap_state.steps < max_iter)

        info_to_be_stored = {'L': L, 'step_size': hyp.step_size, 
                             'EEVPD_wanted': EEVPD_wanted, 'EEVPD': EEVPD, 
                             'equi_diag': equi_diag, 'contracted_exp_vals': contracted_exp_vals}
    
        return AdaptationState(adaptation_state.steps + 1, EEVPD, EEVPD_wanted, hyp), info_to_be_stored
    

