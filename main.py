

RNN_Policy = make_policy_model(features='cnn', policy='lstm')

(input_obs, hidden_states) = generate_bottleneck_data(RNN_Policy)
(input_obs,
 actions,
 action_distributions) = generate_training_traces(RNN_Policy)

obs_qbn_num_quantized_obs = 64
observation_QBN = QBN(io_size=observation_dims,
                      quantized_size=obs_qbn_num_quantized_obs,
                      hyperparams=obs_qbn_hyperparams)

hid_state_num_quantized_states = 100
hidden_state_QBN = QBN(io_size=hidden_state_dims,
                       quantized_size=hid_state_num_quantized_states,)
hidden_state_QBN.train(hidden_states, hyperparams=hid_state_qbn_hyperparams)

moore_machine_net = build_MMN(RNN_Policy, observation_QBN, hidden_state_QBN)
mmn_training_data = (input_obs, actions, action_distributions)
moore_machine_net.train(mmn_training_data,
                        hyperparams=mmn_training_hyperparams)

classical_mm = moore_machine()
classical_mm.extract_from_nn(env, moore_machine_net)
classical_mm.minimize(moore_machine_net)

