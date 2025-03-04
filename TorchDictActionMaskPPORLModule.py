import functools

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.utils import override
from ray.tune.registry import register_env

import gymnasium as gym
import numpy as np

from typing import Any, Dict, Optional
from ray.rllib.utils.typing import TensorType

from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN


from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI


torch, nn = try_import_torch()


class MaskedTorchPPORLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        assert isinstance(self.action_space, gym.spaces.Discrete)
        assert isinstance(self.observation_space, gym.spaces.Dict)

        self.concat_embedding_size = 32
#         self.pi_out_features = np.prod(self.action_space.nvec)
        
        self.pi_out_features = self.action_space.n
        self.observation = self.observation_space['observations']
        
        self.action_mask = self.observation_space['action_mask']
        

#         self.observation_encoder = nn.Sequential(
#             nn.Linear(in_features=self.observation.shape[0], out_features=64, bias=True),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=self.observation.shape[0], bias=True),
#         )

#         self.bar_encoder = nn.Sequential(
#             nn.Linear(in_features=self.observation_space["bar"].shape[0], out_features=16, bias=True),
#             nn.ReLU(),
#             nn.Linear(in_features=16, out_features=self.concat_embedding_size, bias=True),
#         )


        self.my_pi = nn.Sequential(
            nn.Linear(in_features=self.observation._shape[0], out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.pi_out_features, bias=True),
        )

        self.my_vf = nn.Sequential(
            nn.Linear(in_features=self.observation._shape[0], out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
        )

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Default forward pass (used for inference and exploration)."""
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
            # Extract the available actions tensor from the observation.
        action_mask = batch[Columns.OBS]["action_mask"]
        obs_data = batch[Columns.OBS]["observations"]
        
        logits = self.my_pi(obs_data)

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        
        # Apply action mask
        masked_logits = logits + inf_mask
        
#         print (logits, masked_logits)
        
        # Return features and logits as ACTION_DIST_INPUTS (categorical distribution).
        return {
            Columns.ACTION_DIST_INPUTS: masked_logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
            
        action_mask = batch[Columns.OBS]["action_mask"]
        obs_data = batch[Columns.OBS]["observations"]
        
        embeddings = obs_data
        logits = self.my_pi(obs_data)
        
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        
        # Apply action mask to logits
        masked_logits = logits + inf_mask
        
        # Return features and logits as ACTION_DIST_INPUTS.
        return {
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.EMBEDDINGS: embeddings,
        }
      
    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
    
        action_mask = batch[Columns.OBS]["action_mask"]
        obs_data = batch[Columns.OBS]["observations"]

         
        return self.my_vf(obs_data).squeeze(-1)  # Squeeze out last dimension (single node value head).
    

class MaskedTorchPPORLModuleCatalog(PPOCatalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        # Shortcircuit completely the normal Catalog __init__ to avoid the call to _get_encoder_config() 
        # which is not implemented yet for Dict spaces
        self.observation_space = observation_space
        self.action_space = action_space

        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space, action_space=self.action_space
        )

    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        **kwargs,
    ):
        if isinstance(observation_space, gym.spaces.Dict):
            # TODO here
            raise NotImplementedError
        else:
            return super()._get_encoder_config(observation_space, **kwargs)


