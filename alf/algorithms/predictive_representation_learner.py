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

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, TimeStep, Experience, LossInfo, namedtuple
from alf.experience_replayers.replay_buffer import BatchInfo, ReplayBuffer
from alf.nest.utils import convert_device
from alf.networks import Network, LSTMEncodingNetwork
from alf.utils.losses import element_wise_squared_loss, element_wise_huber_loss
from alf.utils import dist_utils, spec_utils, tensor_utils
from alf.utils.summary_utils import safe_mean_hist_summary, safe_mean_summary

PredictiveRepresentationLearnerInfo = namedtuple(
    'PredictiveRepresentationLearnerInfo',
    [
        # actual actions taken in the next unroll_steps
        # [B, unroll_steps, ...]
        'action',

        # The flag to indicate whether to include this target into loss
        # [B, unroll_steps + 1]
        'mask',

        # nest for targets
        # [B, unroll_steps + 1, ...]
        'target'
    ])


@gin.configurable
class SimpleDecoder(Algorithm):
    def __init__(self,
                 input_tensor_spec,
                 target_field,
                 decoder_net_ctor,
                 summarize_each_dimension=False,
                 debug_summaries=False,
                 name="SimpleDecoder"):
        super().__init__(debug_summaries=debug_summaries, name=name)
        self._decoder_net = decoder_net_ctor(
            input_tensor_spec=input_tensor_spec)
        assert self._decoder_net.state_spec == (
        ), "RNN decoder is not suppported"
        self._summarize_each_dimension = summarize_each_dimension
        self._target_field = target_field

    def get_target_fields(self):
        return self._target_field

    def train_step(self, repr, state=()):
        predicted_reward = self._decoder_net(repr)[0]
        return AlgStep(
            output=predicted_reward, state=state, info=predicted_reward)

    def calc_loss(self, target, predicted, mask=None):
        """
        Args:
            target: [T, B, ...]
            predicted: [T, B, ...]
            mask: [T, B]
        Returns:
            LossInfo
        """
        loss = element_wise_huber_loss(target, predicted)
        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):

                def _summarize1(pred, tgt, loss, mask, suffix):
                    alf.summary.scalar(
                        "explained_variance" + suffix,
                        tensor_utils.explained_variance(pred, tgt, mask))
                    safe_mean_hist_summary('predict' + suffix, pred, mask)
                    safe_mean_hist_summary('target' + suffix, tgt, mask)
                    safe_mean_summary("loss" + suffix, loss, mask)

                def _summarize(pred, tgt, loss, mask, suffix):
                    _summarize1(pred[0], tgt[0], loss[0], mask[0],
                                suffix + "/current")
                    _summarize1(pred[1:], tgt[1:], loss[1:], mask[1:],
                                suffix + "/future")

                if loss.ndim == 2:
                    _summarize(predicted, target, loss, mask, '')
                elif not self._summarize_each_dimension:
                    m = mask
                    if m is not None:
                        m = m.unsqueeze(-1).expand_as(predicted)
                    _summarize(predicted, target, loss, m, '')
                else:
                    for i in range(predicted.shape[2]):
                        suffix = '/' + str(i)
                        _summarize(predicted[..., i], target[..., i],
                                   loss[..., i], mask, suffix)

        if loss.ndim == 3:
            loss = loss.mean(dim=2)

        if mask is not None:
            loss = loss * mask

        return LossInfo(loss=loss)


@gin.configurable
class PredictiveRepresentationLearner(Algorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 num_unroll_steps,
                 decoder_ctor,
                 encoding_net_ctor,
                 dynamics_net_ctor,
                 debug_summaries=False,
                 name="PredictiveRepresentationLearner"):
        """
        Args:
            decoder_ctor: ``decoder_ctor(observation)``
            encoding_net_ctor: ``encoding_net_ctor(observation_spec)``
            dynamics_net_ctor: ``dynamics_net_ctor(action_spec)``
        """

        encoding_net = encoding_net_ctor(observation_spec)
        super().__init__(
            train_state_spec=encoding_net.state_spec,
            debug_summaries=debug_summaries,
            name=name)

        self._encoding_net = encoding_net
        repr_spec = self._encoding_net.output_spec
        self._dynamics_net = dynamics_net_ctor(action_spec)
        self._decoder = decoder_ctor(
            repr_spec, debug_summaries=debug_summaries, name=name + ".decoder")
        assert len(alf.nest.flatten(self._decoder.train_state_spec)) == 0, (
            "RNN decoder is not suported")
        self._num_unroll_steps = num_unroll_steps
        self._target_fields = self._decoder.get_target_fields()
        self._output_spec = repr_spec
        self._dynamics_state_dims = alf.nest.map_structure(
            lambda spec: spec.numel,
            alf.nest.flatten(self._dynamics_net.state_spec))
        assert sum(
            self._dynamics_state_dims) > 0, ("dynamics_net should be RNN")
        compatible_state = True
        try:
            alf.nest.assert_same_structure(self._dynamics_net.state_spec,
                                           self._encoding_net.state_spec)
            compatible_state = all(
                alf.nest.flatten(
                    alf.nest.map_structure(lambda s1, s2: s1 == s2,
                                           self._dynamics_net.state_spec,
                                           self._encoding_net.state_spec)))
        except Exception:
            compatible_state = False
        self._latent_to_dstate_fc = None
        if not compatible_state:
            self._latent_to_dstate_fc = alf.layers.FC(
                repr_spec.numel, sum(self._dynamics_state_dims))

    @property
    def output_spec(self):
        return self._output_spec

    def predict_step(self, time_step: TimeStep, state):
        latent, state = self._encoding_net(time_step.observation, state)
        return AlgStep(output=latent, state=state)

    def rollout_step(self, time_step: TimeStep, state):
        latent, state = self._encoding_net(time_step.observation, state)
        return AlgStep(output=latent, state=state)

    def train_step(self, exp: TimeStep, state):
        batch_size = exp.step_type.shape[0]
        latent, state = self._encoding_net(exp.observation, state)
        # [B, num_unroll_steps + 1]
        info = exp.rollout_info

        if self._latent_to_dstate_fc is not None:
            dstate = self._latent_to_dstate_fc(latent)
            dstate = dstate.split(self._dynamics_state_dims, dim=1)
            dstate = alf.nest.pack_sequence_as(self._dynamics_net.state_spec,
                                               dstate)
        else:
            dstate = state

        sim_latents = [latent]
        for i in range(self._num_unroll_steps):
            sim_latent, dstate = self._dynamics_net(info.action[:, i, ...],
                                                    dstate)
            sim_latents.append(sim_latent)

        sim_latent = torch.cat(sim_latents, dim=0)

        # [num_unroll_steps + 1)*B, ...]
        train_info = self._decoder.train_step(sim_latent).info
        train_info_spec = dist_utils.extract_spec(train_info)
        train_info = dist_utils.distributions_to_params(train_info)
        train_info = alf.nest.map_structure(
            lambda x: x.reshape(self._num_unroll_steps + 1, batch_size, *x.
                                shape[1:]), train_info)
        # [num_unroll_steps + 1, B, ...]
        train_info = dist_utils.params_to_distributions(
            train_info, train_info_spec)
        target = alf.nest.map_structure(lambda x: x.transpose(0, 1),
                                        info.target)
        loss_info = self._decoder.calc_loss(target, train_info, info.mask.t())
        loss_info = alf.nest.map_structure(lambda x: x.mean(dim=0), loss_info)

        return AlgStep(output=latent, state=state, info=loss_info)

    @torch.no_grad()
    def preprocess_experience(self, experience: Experience):
        """Fill experience.rollout_info with PredictiveRepresentationLearnerInfo

        Note that the shape of experience is [B, T, ...]
        """
        assert experience.batch_info != ()
        batch_info: BatchInfo = experience.batch_info
        replay_buffer: ReplayBuffer = experience.replay_buffer
        mini_batch_length = experience.step_type.shape[1]

        with alf.device(replay_buffer.device):
            # [B, 1]
            positions = convert_device(batch_info.positions).unsqueeze(-1)
            # [B, 1]
            env_ids = convert_device(batch_info.env_ids).unsqueeze(-1)

            # [B, T]
            positions = positions + torch.arange(mini_batch_length)

            # [B, T]
            steps_to_episode_end = replay_buffer.steps_to_episode_end(
                positions, env_ids)
            # [B, T]
            episode_end_positions = positions + steps_to_episode_end

            # [B, T, unroll_steps+1]
            positions = positions.unsqueeze(-1) + torch.arange(
                self._num_unroll_steps + 1)
            # [B, 1, 1]
            env_ids = env_ids.unsqueeze(-1)
            # [B, T, 1]
            episode_end_positions = episode_end_positions.unsqueeze(-1)

            # [B, T, unroll_steps+1]
            mask = positions <= episode_end_positions

            # [B, T, unroll_steps+1]
            positions = torch.min(positions, episode_end_positions)

            # [B, T, unroll_steps+1]
            target = replay_buffer.get_field(self._target_fields, env_ids,
                                             positions)

            # [B, T, unroll_steps]
            action = replay_buffer.get_field('action', env_ids,
                                             positions[:, :, :-1])

            rollout_info = PredictiveRepresentationLearnerInfo(
                action=action, mask=mask, target=target)

        rollout_info = convert_device(rollout_info)

        return experience._replace(rollout_info=rollout_info)
