class Entity(object):
    def __init__(self, agents_args_list, entity_args):
        self.agents = []
        for args in agents_args_list:
            agent = Agent(algorithm=args["algorithm"],
                          discount=args["discount"],
                          snapshot=args["snapshot"],
                          max_memory=args["max_memory"],
                          prioritized_experience=args["prioritized_experience"],
                          exploration_policy=args["exploration_policy"],
                          learning_rate=args["learning_rate"],
                          level=args["level"],
                          history_length=args["history_length"],
                          batch_size=args["batch_size"],
                          temperature=args["temperature"],
                          combine_actions=args["combine_actions"],
                          train=(args["mode"] == Mode.TRAIN),
                          skipped_frames=args["skipped_frames"],
                          visible=False,
                          target_update_freq=args["target_update_freq"],
                          epsilon_start=args["epsilon_start"],
                          epsilon_end=args["epsilon_end"],
                          epsilon_annealing_steps=args["epsilon_annealing_steps"])

            if (args["mode"] == Mode.TEST or args["mode"] == Mode.DISPLAY) and args["snapshot"] == '':
                print("Warning: mode set to " + str(args["mode"]) + " but no snapshot was loaded")

            self.agents += [agent]
        self.episodes = entity_args["episodes"]
        self.steps_per_episode = entity_args["steps_per_episode"]
        self.mode = entity_args["mode"]
        self.start_learning_after = entity_args["start_learning_after"]
        self.average_over_num_episodes = entity_args["average_over_num_episodes"]
        self.snapshot_episodes = entity_args["snapshot_episodes"]
        self.environment = Environment(level=entity_args["level"], combine_actions=entity_args["combine_actions"])
        self.history_length = entity_args["history_length"]
        self.win_count = 0
        self.curr_step = 0

    def combine_actions(self, aiming_actions, exploring_actions):
        # aiming_actions (defend_the_center) = 1. TURN_LEFT, 2. TURN_RIGHT, 3. ATTACK
        # exploring_actions (health_gathering or my_way_home) = 1. TURN_LEFT, 2. TURN_RIGHT, 3. MOVE_FORWARD, 4. MOVE_LEFT, 5. MOVE_RIGHT
        # death match actions (deathmatch) = 1. ATTACK, 2. SPEED, 3. STRAFE, 4. MOVE_RIGHT, 5. MOVE_LEFT, 6. MOVE_BACKWARD, 7. MOVE_FORWARD,
        #                       8. TURN_RIGHT, 9. TURN_LEFT, 10. SELECT_WEAPON1, 11. SELECT_WEAPON2, 12. SELECT_WEAPON3,
        #                       13. SELECT_WEAPON4, 14. SELECT_WEAPON5, 15. SELECT_WEAPON6, 16. SELECT_NEXT_WEAPON,
        #                       17. SELECT_PREV_WEAPON, 18. LOOK_UP_DOWN_DELTA, 19. TURN_LEFT_RIGHT_DELTA, 20. MOVE_LEFT_RIGHT_DELTA

        actions = [False] * 20
        actions[0] = aiming_actions[2]      # attack
        actions[3] = exploring_actions[4]   # move right
        actions[4] = exploring_actions[3]   # move left
        actions[6] = exploring_actions[2]   # move forward
        actions[7] = aiming_actions[1] or exploring_actions[1]  # turn right
        actions[8] = aiming_actions[0] or exploring_actions[0]  # turn left
        actions[11] = True # always use gun

        return actions

    def step(self, action):
        # repeat action several times and stack the states
        reward = 0
        game_over = False
        next_state = list()
        for t in range(self.history_length):
            s, r, game_over = self.environment.step(action)
            reward += r # reward is accumulated
            if game_over:
                break
            next_state.append(s)

        # episode finished
        if game_over:
            self.environment.new_episode()

        if reward > 0 and game_over:
            self.win_count += 1

        return next_state, reward, game_over

    def run(self):
        # initialize
        total_steps, average_return = 0, 0
        returns = []
        for i in range(self.episodes):
            self.environment.new_episode()
            steps, curr_return = 0, 0
            game_over = False
            while not game_over and steps < self.steps_per_episode:
                # each agent predicts the action it should do
                actions, action_idxs = [], []
                for agent in self.agents:
                    action, action_idx, _ = agent.predict()
                    actions += [action]
                    action_idxs += [action_idx]
                # the actions are combined together
                action = self.combine_actions(actions[0], actions[1]) #TODO: make this more generic
                # the entity performs the action
                next_state, reward, game_over = self.step(action)
                # each agent preprocesses the next state and stores it
                for agent_idx, agent in enumerate(self.agents):
                    agent.store_next_state(next_state, reward, game_over, action_idxs[agent_idx])

                steps += 1
                curr_return += reward

                # delay a bit so we humans can understand what we are seeing
                if self.mode == Mode.DISPLAY:
                    sleep(0.05)

                if i > self.start_learning_after and self.mode == Mode.TRAIN:
                    for agent in self.agents:
                        agent.train()

            # average results
            n = float(self.average_over_num_episodes)
            average_return = (1 - 1 / n) * average_return + (1 / n) * curr_return
            total_steps += steps
            returns += [average_return]

            # print progress
            print("")
            print(str(datetime.datetime.now()))
            print("episode = " + str(i) + " steps = " + str(total_steps))
            print("current_return = " + str(curr_return) + " average return = " + str(average_return))

            # save snapshot of target network
            if i % self.snapshot_episodes == self.snapshot_episodes - 1:
                for agent_idx, agent in enumerate(self.agents):
                    snapshot = 'agent' + str(agent_idx) + '_model_' + str(i + 1) + '.h5'
                    snapshot = resultDir + "models/" + snapshot
                    print(str(datetime.datetime.now()) + " >> saving snapshot to " + snapshot)
                    agent.target_network.save_weights(snapshot, overwrite=True)

        self.environment.game.close()
        return returns
