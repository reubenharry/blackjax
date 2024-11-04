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

from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp

from blackjax.util import run_eca
from blackjax.mcmc.integrators import isokinetic_mclachlan, isokinetic_velocity_verlet
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.mhmclmc import build_kernel_malt
import blackjax.adaptation.ensemble_umclmc as umclmc
from blackjax.adaptation.ensemble_umclmc import equipartition_diagonal, equipartition_diagonal_loss, equipartition_fullrank, equipartition_fullrank_loss

from blackjax.adaptation.step_size import dual_averaging_adaptation

    

class AdaptationState(NamedTuple):
    steps_per_sample: float
    step_size: float
    da_state: Any
    


def build_kernel(logdensity_fn, integrator):
    """MCLMC kernel"""
    
    kernel = build_kernel_malt(logdensity_fn, integrator)
    
    def sequential_kernel(key, state, adap):
        return kernel(key, state, step_size= adap.step_size, num_integration_steps= adap.steps_per_sample)
    
    return sequential_kernel



class Adaptation:
    
    def __init__(self, adap_state, 
                 num_dims, 
                 mclachlan, steps_per_sample, acc_prob_target= 0.8,
                 monitor_exp_vals= lambda x: 0., contract_exp_vals= lambda exp_vals: 0.):
        
        self.monitor_exp_vals = monitor_exp_vals
        self.contract_exp_vals = contract_exp_vals
        
        ### Determine the initial hyperparameters ###
        
        ## stepsize ##
        #if we switched to the more accurate integrator we can use longer step size
        integrator_factor = jnp.sqrt(10.) if mclachlan else 1. 
        # Let's use the stepsize which will be optimal for the adjusted method. The energy variance after N steps scales as sigma^2 ~ N^2 eps^6 = eps^4 L^2
        # In the adjusted method we want sigma^2 = 2 mu = 2 * 0.41 = 0.82
        # With the current eps, we had sigma^2 = EEVPD * d for N = 1. 
        # Combining the two we have EEVPD * d / 0.82 = eps^6 / eps_new^4 L^2
        adjustment_factor = jnp.power(0.82 / (num_dims * adap_state.EEVPD), 0.25) / jnp.sqrt(steps_per_sample)
        
        step_size = adap_state.step_size * integrator_factor * adjustment_factor

        #steps_per_sample = (int)(jnp.max(jnp.array([Lfull / step_size, 1])))
                
        
        ### Initialize the dual averaging adaptation ###
        da_init, self.da_update, da_final = dual_averaging_adaptation(target= acc_prob_target)
        
        da_state = da_init(step_size)
        
        self.initial_state = AdaptationState(steps_per_sample, step_size, da_state)
        
        
    def summary_statistics_fn(self, state, info):
     
        return {'acceptance_probability': info.acceptance_rate,
                'equipartition_diagonal': equipartition_diagonal(state), 
                'equipartition_fullrank': equipartition_fullrank(state), 
                'monitored_exp_vals': self.monitor_exp_vals(state.position)
                }
        
        
    def update(self, adaptation_state, Etheta, key_adaptation):
        
        # combine the expectation values to get useful scalars
        acc_prob = Etheta['acceptance_probability']
        equi_diag = equipartition_diagonal_loss(Etheta['equipartition_diagonal'])
        equi_full = equipartition_fullrank_loss(Etheta['equipartition_fullrank'])
        contracted_exp_vals = self.contract_exp_vals(Etheta['monitored_exp_vals'])
        
        
        info_to_be_stored = {'L': adaptation_state.step_size * adaptation_state.steps_per_sample, 'steps_per_sample': adaptation_state.steps_per_sample, 'step_size': adaptation_state.step_size, 
                             'acc_prob': acc_prob,
                             'equi_diag': equi_diag, 'equi_full': equi_full, 'contracted_exp_vals': contracted_exp_vals}


        # hyperparameter adaptation                                              
        da_state = self.da_update(adaptation_state.da_state, acc_prob)
        step_size = jnp.exp(da_state.log_step_size)
        
        return AdaptationState(adaptation_state.steps_per_sample, step_size, da_state), info_to_be_stored



def bias(model):
    
    def exp_vals(position):
        return jnp.square(model.transform(position))
    
    def contract(sampler_E_x2):
        bsq = jnp.square(sampler_E_x2 - model.E_x2) / model.Var_x2
        return jnp.array([jnp.max(bsq), jnp.average(bsq)])
    
    return exp_vals, contract



def emaus(model, num_steps1, num_steps2, num_chains, mesh, rng_key,
          mclachlan= True, steps_per_sample= 10, acc_prob_target= 0.8):
    
    exp_vals, contract = bias(model)
    key_init, key_umclmc, key_mclmc = jax.random.split(rng_key, 3)
    
    # initialize the chains
    initial_state = umclmc.initialize(key_init, model.logdensity_fn, model.sample_init, num_chains, mesh)
    
    ### burn-in with the unadjusted method ###
    kernel = umclmc.build_kernel(model.logdensity_fn)
    adap = umclmc.Adaptation(model.ndims, monitor_exp_vals= exp_vals, contract_exp_vals= contract)
    final_state, final_adaptation_state, info1 = run_eca(key_umclmc, initial_state, kernel, adap, num_steps1, num_chains, mesh)
    
    
    ### refine the results with the adjusted method ###
    integrator, gradient_calls_per_step = (isokinetic_mclachlan, 2) if mclachlan else (isokinetic_velocity_verlet, 1)
    kernel = build_kernel(model.logdensity_fn, integrator)
    initial_state= HMCState(final_state.position, final_state.logdensity, final_state.logdensity_grad)    

    adap = Adaptation(final_adaptation_state, model.ndims, 
                      mclachlan, steps_per_sample, acc_prob_target, 
                      monitor_exp_vals= exp_vals, contract_exp_vals= contract)
    
    num_samples = num_steps2 // (gradient_calls_per_step * steps_per_sample)
    final_state, final_adaptation_state, info2 = run_eca(key_mclmc, initial_state, kernel, adap, num_samples, num_chains, mesh)
    
    return info1, info2
    
    
