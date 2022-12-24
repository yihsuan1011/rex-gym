# Use ToolBox to call all of tools

from . import wrappers

from .attr_dict import AttrDict
from .batch_env import BatchEnv
from .in_graph_batch_env import InGraphBatchEnv
from .loop import Loop

from .count_weights import count_weights
from .simulate import simulate

class Toolbox:
    def __init__(self):
        pass
    
    def _attr_dict(args, kwargs):
        return AttrDict(args, kwargs)
    
    def _batch_env(envs, blocking):
        return BatchEnv(envs, blocking)
    
    def _in_graph__batch_env(batch_env):
        return InGraphBatchEnv(batch_env)
    
    def _loop(logdir, step, log, report, reset):
        return Loop(logdir, step, log, report, reset)
        
    def _external_process(constructor):
        return wrappers.ExternalProcess(constructor)
    
    def _limit_duration(env, duration):
        return wrappers.LimitDuration(env, duration)
    
    def _range_normalize(env):
        return wrappers.RangeNormalize(env)
    
    def _clip_action(env):
        return wrappers.ClipAction(env)
    
    def _convert_to_32_bit(env):
        return wrappers.ConvertTo32Bit(env)
    
    def _count_weights():
        count_weights()
    
    def _simulate(batch_env, algo, log, reset):
        simulate(batch_env, algo, log, reset)
