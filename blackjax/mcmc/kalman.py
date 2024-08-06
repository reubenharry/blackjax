
from typing import Callable, NamedTuple, Any



class KalmanState(NamedTuple):

    avg_observable: Any
    weight: int
    dynamic: Any 
    
       

def kalmanize(inference_algorithm, observable):
    
    def new_init(init_position, key):
        dynamic_state = inference_algorithm.init(init_position, key)
        obs = observable(dynamic_state)
        
        return KalmanState(obs, 1., dynamic_state)
    
        
    def new_kernel(state):
        new_dynamic_state, info = inference_algorithm.kernel(state.dynamic)
        obs = observable(new_dynamic_state)
        
        return KalmanState(
            (state.avg_observable * state.weight + obs) / (state.weight + 1), 
            state.weight + 1, 
            new_dynamic_state), info
        
