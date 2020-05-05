def generate_bottleneck_data(policy_network, env, traces_output_path
                             epsilon_greedy_prob=0.3,
                             max_generation_steps=100
                             num_episodes=20):

    bottleneck_data = {}
    hx_data, obs_data, action_data = [], [], []
    all_ep_rewards = []
    with torch.no_grad():
        for ep in range(episodes):

            done = False
            obs = env.reset()
            hx = policy_network.initial_state
            print(hx)
            ep_reward = 0
            act_count = 0
            max_steps_to_explore_over = random.choice(range(0, max_generation_steps, 
                int(0.02 * max_generation_steps)))

            while not done:
                env.render()
                obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                if cuda:
                    hx = hx.cuda()
                    obs = obs.cuda()
                
                critic, logit, hx, (_, _, obs_c, _) = policy_network((obs, hx), inspect=True)
                
                should_explore = (random.random() < epsilon_greedy_prob[ep %    
                    len(epsilon_greedy_prob)])
                exploration_start_step >= act_count
                take_random_act = () and 
                                  
                if take_random_act:
                    action = env.action_space.sample()
                else:
                    prob = F.softmax(logit, dim=1)
                    action = int(prob.max(1)[1].data.cpu().numpy())
                obs, reward, done, info = env.step(action)
                action_data.append(action)
                act_count += 1
                done = done if act_count <= max_generation_steps else True
                if action not in bottleneck_data:
                    bottleneck_data[action] = {'hx_data': [], 'obs_data': []}
                bottleneck_data[action]['hx_data'].append(hx.data.cpu().numpy()[0].tolist())
                bottleneck_data[action]['obs_data'].append(obs_c.data.cpu().numpy()[0].tolist())

                ep_reward += reward
            logging.info('episode:{} reward:{}'.format(ep, ep_reward))
            all_ep_rewards.append(ep_reward)
    logging.info('Average Performance:{}'.format(sum(all_ep_rewards) / len(all_ep_rewards)))

    hx_train_data, hx_test_data, obs_train_data, obs_test_data = [], [], [], []
    for action in bottleneck_data.keys():
        hx_train_data += bottleneck_data[action]['hx_data']
        hx_test_data += bottleneck_data[action]['hx_data']
        obs_train_data += bottleneck_data[action]['obs_data']
        obs_test_data += bottleneck_data[action]['obs_data']

        logging.info('Action: {} Hx Data: {} Obs Data: {}'.format(action, len(np.unique(bottleneck_data[action]['hx_data'], axis=0).tolist()), len(np.unique(bottleneck_data[action]['obs_data'], axis=0).tolist())))

    obs_test_data = np.unique(obs_test_data, axis=0).tolist()
    hx_test_data = np.unique(hx_test_data, axis=0).tolist()

    random.shuffle(hx_train_data)
    random.shuffle(obs_train_data)
    random.shuffle(hx_test_data)
    random.shuffle(obs_test_data)

    pickle.dump((hx_train_data, hx_test_data, obs_train_data, obs_test_data), open(save_path, "wb"))

logging.info('Data Sizes:')
logging.info('Hx Train:{} Hx Test:{} Obs Train:{} Obs Test:{}'.format(len(hx_train_data), len(hx_test_data), len(obs_train_data), len(obs_test_data)))

return hx_train_data, hx_test_data, obs_train_data, obs_test_data

return (input_obs, hidden_states)

generate_bottleneck_data(basepolicy_model,
                         environment=env,
                         traces_output_path=BASEPOLICY_TRACES_PATH)