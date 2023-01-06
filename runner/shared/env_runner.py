
import time
import numpy as np
import torch
from runner.shared.base_runner import Runner
import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        # reset环境并将obs和share_obs存入buffer
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # 训练episodes次
        for episode in range(episodes):
            # 更新learning rate
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # 每个episode有self.episode_length步
            for step in range(self.episode_length):
                # 通过该step的观测值得到action、适应环境的action_env和它的概率，以及该状态的value
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # 环境采取动作得到next_state、reward、dones
                obs, rewards, dones, infos = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                # 将数据存入buffer
                self.insert(data)

            # 在buffer中根据reward计算每一步的returns (Q value)
            self.compute()

            # 更新网络并返回loss值等
            train_infos = self.train()

            # 已完成的所有step数之和
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n {}/{} episodes, {}/{} steps, {}/{} seconds.\n"
                      .format(episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(end - start),
                              int(self.num_env_steps / total_num_steps * (end - start))))
                # buffer中的episode的总reward
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)

            # evaluate
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.episode_length)

    def warmup(self):
        obs = self.envs.reset()  # shape = (并行环境数, 智能体数, 观测环境维数)
        # 产生share_obs,实现去重
        share_obs = self.generate_share_obs(obs)
        # 将obs和share_obs存入buffer
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # 将该step的观测值输入actor网络得到action和它的概率，并且用critic网络得到该动状态的value
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # 对数据进行整形，shape=[envs_num, agent_num, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        """actions范围是无穷，需要改成适合环境的"""
        actions_env = actions.clip(-1, 1)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # 产生share_obs，实现去重
        share_obs = self.generate_share_obs(obs)

        # 将数据存到buffer中（这里的obs是下一时刻的obs了）
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks)

    def generate_share_obs(self, obs):
        """"""
        """定义share_obs，实现去重的工作（有两处用到）"""
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = (并行环境数, 所有agent观测维数之和/公共维数)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # shape = (并行环境数, agent个数, 所有agent观测维数之和)
        else:
            share_obs = obs
        return share_obs

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            # 得到动作，deterministic=True时输出均值，不采样
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            """actions范围是无穷，需要改成适合环境的"""
            eval_actions_env = eval_actions.clip(-1, 1)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        # 每一步的reward
        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_env_infos = {}
        # 该episode总的reward, shape = [并行环境数, agent数, 1]
        eval_env_infos['eval_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        # 对每个环境、每个agent做平均
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

        """画图"""
        self.eval_envs.render()
