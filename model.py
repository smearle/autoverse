from pdb import set_trace as TT

from einops import rearrange
import torch as th
from torch import nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override


class CustomFeedForwardModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=64,
                 fc_size=64,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape
        self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters
        self.fc_size = fc_size

        # TODO: use more convolutions here? Change and check that we can still overfit on binary problem.
        self.conv_1 = nn.Conv2d(obs_space.shape[-1], out_channels=conv_filters, kernel_size=3, stride=1, padding=0)

        self.fc_1 = SlimFC(self.pre_fc_size, self.fc_size)
        self.action_branch = SlimFC(self.fc_size, num_outputs)
        self.value_branch = SlimFC(self.fc_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        input = input_dict["obs"].permute(0, 3, 1, 2)  # Because rllib order tensors the tensorflow way (channel last)
        x = nn.functional.relu(self.conv_1(input.float()))
        x = x.reshape(x.size(0), -1)
        x = nn.functional.relu(self.fc_1(x))
        self._features = x
        action_out = self.action_branch(self._features)

        return action_out, []


class TorchCustomModel(TorchModelV2, nn.Module):
    """Custom model."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(
            in_channels=obs_space.shape[2],
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.flat_size = 32 * obs_space.shape[0] * obs_space.shape[1]
        print(self.flat_size)
        TT()  #TODO: need to alter the obs_space here...
        # self.linear = nn.Linear(self.flat_size, num_outputs)
        self.fc = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        obs = rearrange(obs, "b h w c -> b c h w")
        conv_out = th.relu(self.conv(obs))
        print(conv_out.shape)
        TT()
        conv_out_flat = rearrange(conv_out, "b c h w -> b (c h w)")
        fc_out, _ = self.fc(
            input_dict={
                "obs": conv_out, 
                "obs_flat": conv_out_flat,
            }, 
            state=state, 
            seq_lens=seq_lens
        )
        return fc_out, []

    def value_function(self):
        return th.reshape(self.fc.value_function(), [-1])

