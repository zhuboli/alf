# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gin
import torch

# implement the respective reward functions for desired environments here


@gin.configurable
def reward_function_for_pendulum(obs, action):
    """Function for computing reward for gym Pendulum environment. It takes
        as input:
        (1) observation (Tensor of shape [batch_size, observation_dim])
        (2) action (Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].

        Note that in the planning module, (next_obs, action) is currently used
        as the input to this function. Might need to consider (obs, action)
        on the caller side in order to be more compatible with the conventional
        definition of the reward function.
    """

    def _observation_cost(obs):
        c_theta, s_theta, d_theta = obs[:, :1], obs[:, 1:2], obs[:, 2:3]
        theta = torch.atan2(s_theta, c_theta)
        cost = theta**2 + 0.1 * d_theta**2
        cost = torch.sum(cost, dim=1)
        cost = torch.where(
            torch.isnan(cost), 1e6 * torch.ones_like(cost), cost)
        return cost

    def _action_cost(action):
        return 0.001 * torch.sum(action**2, dim=1)

    cost = _observation_cost(obs) + _action_cost(action)
    # negative cost as reward
    reward = -cost
    return reward
