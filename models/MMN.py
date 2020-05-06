import numpy as np
import gym


class MMNet:
    """
    Moore Machine Network(MMNet) definition.

    basic idea from:
    https://github.com/koulanurag/mmn/blob/65135da36dc8f8bfad05797f9d077c22adde0384/main_atari.py#L155
    """

    def __init__(self, rnn_policy, obs_qbn, hidden_state_qbn, env,
                 step_all_vals_wrapper):

        self.obs_qbn = obs_qbn
        self.rnn_policy = rnn_policy
        self.hidden_state_qbn = hidden_state_qbn
        self.env = env
        self.step_all_vals_wrapper = step_all_vals_wrapper
        self.n_envs = rnn_policy.n_envs

    def predict(self, observation, state=None, mask=None, deterministic=False):

        if state is None:
            state = self.rnn_policy.initial_state
        if mask is None:
            mask = [False for _ in range(self.rnn_policy.n_envs)]

        agent = self.rnn_policy
        policy = agent.act_model
        obs_qbn = self.obs_qbn
        hid_state_qbn = self.hidden_state_qbn

        observation = np.array(observation)
        obsv_space = agent.observation_space
        vectorized_env = agent._is_vectorized_observation(observation,
                                                          obsv_space)
        new_obs_shape = (-1,) + agent.observation_space.shape
        observation = observation.reshape(new_obs_shape)

        # need to remove obervations from the vectorized environments;
        # we just care about single-environment performance
        vec_obs_shape = (agent.n_envs,) + agent.observation_space.shape
        zero_completed_obs = np.zeros(vec_obs_shape)
        zero_completed_obs[0, :] = observation

        (_, _, _,
         _, _, _,
         _, _,
         extracted_fs) = self.step_all_vals_wrapper(agent, observation,
                                                    self.env, state, mask,
                                                    deterministic)

        # run the feature through the qbn
        # need to QBN each environment's set of feature separately
        num_envs, num_features = extracted_fs.shape
        extracted_fs_through_qbn = np.zeros(extracted_fs.shape,
                                            dtype=np.float32)
        for idx, feature in enumerate(extracted_fs):
            extracted_f_through_qbn = obs_qbn.decode(obs_qbn.encode(feature))
            extracted_fs_through_qbn[idx, :] = extracted_f_through_qbn

        # now, we need to get the rnn_output given the obs_qbn'd features
        ops = policy.input_sequence
        feed_dict = {policy._extracted_features: extracted_fs_through_qbn,
                     policy.obs_ph: zero_completed_obs,
                     policy.states_ph: state,
                     policy.dones_ph: mask}
        input_sequence_through_qbn = policy.sess.run(ops, feed_dict)

        ops = policy.rnn_output
        feed_dict = {policy.input_sequence[0]: input_sequence_through_qbn[0],
                     policy.obs_ph: zero_completed_obs,
                     policy.states_ph: state,
                     policy.dones_ph: mask}
        rnn_output = policy.sess.run(ops, feed_dict)

        quantized_latent_state = hid_state_qbn.encode(rnn_output)
        rnn_output_through_qbn = hid_state_qbn.decode(quantized_latent_state)

        # now we need the action and the policy state from the hid_state_qbn'd
        # rnn_output
        ops = [policy.action, policy.snew]
        feed_dict = {policy.rnn_output: rnn_output_through_qbn,
                     policy.obs_ph: zero_completed_obs,
                     policy.states_ph: state,
                     policy.dones_ph: mask}
        actions, states = policy.sess.run(ops, feed_dict)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(agent.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, agent.action_space.low,
                                      agent.action_space.high)

        if not vectorized_env:
            if state is not None:
                msg = "Error: The environment must be vectorized when " + \
                      "using recurrent policies."
                raise ValueError(msg)
            clipped_actions = clipped_actions[0]

        return clipped_actions, states
