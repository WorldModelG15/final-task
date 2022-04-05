from datetime import datetime
import time
import os

import numpy as np
from PIL import Image

import torch
from dreamer.config import DreamerConfig
from dreamer.models import Encoder, RSSM, ValueModel, ActionModel
from dreamer.agent import Agent
from dreamer.utils import ReplayBuffer, preprocess_obs, lambda_target
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_


class Trainer:
    def __init__(self, env, device, config: DreamerConfig, train=True):
        # リプレイバッファの宣言
        buffer_capacity = config.buffer_capacity
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            observation_shape=env.observation_space.shape,
            action_dim=env.action_space.shape[0],
        )

        self.env = env
        self.device = device

        # モデルの宣言
        self.state_dim = config.state_dim  # 確率的状態の次元
        self.rnn_hidden_dim = config.rnn_hidden_dim  # 決定的状態（RNNの隠れ状態）の次元

        self.action_dim = env.action_space.shape[0]

        # 確率的状態の次元と決定的状態（RNNの隠れ状態）の次元は一致しなくて良い
        self.encoder = Encoder().to(device)
        self.rssm = RSSM(self.state_dim, self.action_dim, self.rnn_hidden_dim, device)
        self.value_model = ValueModel(self.state_dim, self.rnn_hidden_dim).to(device)
        self.action_model = ActionModel(
            self.state_dim, self.rnn_hidden_dim, self.action_dim
        ).to(device)

        # オプティマイザの宣言
        model_lr = config.model_lr  # encoder, rssm, obs_model, reward_modelの学習率
        value_lr = config.value_lr
        action_lr = config.action_lr
        eps = config.eps
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.rssm.transition.parameters())
            + list(self.rssm.observation.parameters())
            + list(self.rssm.reward.parameters())
        )
        self.model_optimizer = torch.optim.Adam(self.model_params, lr=model_lr, eps=eps)
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(), lr=value_lr, eps=eps
        )
        self.action_optimizer = torch.optim.Adam(
            self.action_model.parameters(), lr=action_lr, eps=eps
        )

        self.log_dir = "runs/" + datetime.now().strftime("%Y%m%d%H%M%S")
        if train and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.model_save_dir = "models/" + datetime.now().strftime("%Y%m%d%H%M%S")
        if train and not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir, exist_ok=True)

        self.gif_dir = "gif/"
        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        # その他ハイパーパラメータ
        self.seed_episodes = config.seed_episodes  # 最初にランダム行動で探索するエピソード数
        self.all_episodes = config.all_episodes  # 学習全体のエピソード数
        self.test_interval = config.test_interval  # 何エピソードごとに探索ノイズなしのテストを行うか
        self.model_save_interval = config.model_save_interval  # NNの重みを何エピソードごとに保存するか
        self.collect_interval = (
            config.collect_interval
        )  # 何回のNNの更新ごとに経験を集めるか（＝1エピソード経験を集めるごとに何回更新するか）

        self.action_noise_var = config.action_noise_var  # 探索ノイズの強さ

        self.batch_size = config.batch_size
        self.chunk_length = config.chunk_length  # 1回の更新で用いる系列の長さ
        self.imagination_horizon = (
            config.imagination_horizon  # Actor-Criticの更新のために, Dreamerで何ステップ先までの想像上の軌道を生成するか
        )

        self.gamma = config.gamma  # 割引率
        self.lambda_ = config.lambda_  # λ-returnのパラメータ
        self.clip_grad_norm = config.clip_grad_norm  # gradient clippingの値
        self.free_nats = (
            config.free_nats  # KL誤差（RSSMのTransitionModelにおけるpriorとposteriorの間の誤差）がこの値以下の場合, 無視する
        )

    def _add_exploration(self, action, iter):
        """
        たまにランダム行動（Epsilon-greedy）
        """
        expl_amount = self.train_noise
        expl_amount = expl_amount - iter / self.expl_decay
        expl_amount = max(self.expl_min, expl_amount)

        if np.random.uniform(0, 1) < expl_amount:
            index = np.random.randint(0, self.action_dim, action.shape[:-1])
            action = np.zeros_like(action)
            action[index] = 1
        return action

    def train(self):
        for episode in range(self.seed_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(obs, action, reward, done)
                obs = next_obs

        steps = 0

        for episode in range(self.seed_episodes, self.all_episodes):
            # -----------------------------
            #      経験を集める
            # -----------------------------
            start = time.time()
            # 行動を決定するためのエージェントを宣言
            policy = Agent(self.encoder, self.rssm.transition, self.action_model)

            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = policy(obs)
                # 探索のためにガウス分布に従うノイズを加える(explaration noise)
                action += np.random.normal(
                    0, np.sqrt(self.action_noise_var), self.env.action_space.shape[0]
                )
                next_obs, reward, done, _ = self.env.step(action)

                # リプレイバッファに観測, 行動, 報酬, doneを格納
                self.replay_buffer.push(obs, action, reward, done)

                obs = next_obs
                total_reward += reward
                steps += 1

            # 訓練時の報酬と経過時間をログとして表示
            self.writer.add_scalar("total reward at train", total_reward, episode)
            print(
                "episode [%4d/%4d] is collected. Total reward is %f"
                % (episode + 1, self.all_episodes, total_reward)
            )
            print("elasped time for interaction: %.2fs" % (time.time() - start))

            # NNのパラメータを更新する
            start = time.time()
            for update_step in range(self.collect_interval):
                # -------------------------------------------------------------------------------------
                #  RSSM(trainsition_model, obs_model, reward_model)の更新 - Dynamics learning
                # -------------------------------------------------------------------------------------
                observations, actions, rewards, _ = self.replay_buffer.sample(
                    self.batch_size, self.chunk_length
                )

                # 観測を前処理し, RNNを用いたPyTorchでの学習のためにTensorの次元を調整
                observations = preprocess_obs(observations)
                observations = torch.as_tensor(observations, device=self.device)
                observations = observations.transpose(0, 1)
                actions = torch.as_tensor(actions, device=self.device).transpose(0, 1)
                rewards = torch.as_tensor(rewards, device=self.device).transpose(0, 1)

                # 観測をエンコーダで低次元のベクトルに変換
                embedded_observations = self.encoder(
                    observations.reshape(-1, *self.env.observation_space.shape)
                ).view(self.chunk_length, self.batch_size, -1)

                # 低次元の状態表現を保持しておくためのTensorを定義
                states = torch.zeros(
                    self.chunk_length,
                    self.batch_size,
                    self.state_dim,
                    device=self.device,
                )
                rnn_hiddens = torch.zeros(
                    self.chunk_length,
                    self.batch_size,
                    self.rnn_hidden_dim,
                    device=self.device,
                )

                # 低次元の状態表現は最初はゼロ初期化（timestep１つ分）
                state = torch.zeros(self.batch_size, self.state_dim, device=self.device)
                rnn_hidden = torch.zeros(
                    self.batch_size, self.rnn_hidden_dim, device=self.device
                )

                # 状態s_tの予測を行ってそのロスを計算する（priorとposteriorの間のKLダイバージェンス）
                kl_loss = 0
                for l in range(self.chunk_length - 1):
                    (
                        next_state_prior,
                        next_state_posterior,
                        rnn_hidden,
                    ) = self.rssm.transition(
                        state, actions[l], rnn_hidden, embedded_observations[l + 1]
                    )
                    state = next_state_posterior.rsample()
                    states[l + 1] = state
                    rnn_hiddens[l + 1] = rnn_hidden
                    kl = kl_divergence(next_state_prior, next_state_posterior).sum(
                        dim=1
                    )
                    kl_loss += kl.clamp(
                        min=self.free_nats
                    ).mean()  # 原論文通り, KL誤差がfree_nats以下の時は無視
                kl_loss /= self.chunk_length - 1

                # states[0] and rnn_hiddens[0]はゼロ初期化なので以降では使わない
                # states, rnn_hiddensは低次元の状態表現
                states = states[1:]
                rnn_hiddens = rnn_hiddens[1:]

                # 観測を再構成, また, 報酬を予測
                flatten_states = states.view(-1, self.state_dim)
                flatten_rnn_hiddens = rnn_hiddens.view(-1, self.rnn_hidden_dim)
                recon_observations = self.rssm.observation(
                    flatten_states, flatten_rnn_hiddens
                ).view(
                    self.chunk_length - 1,
                    self.batch_size,
                    *self.env.observation_space.shape
                )
                predicted_rewards = self.rssm.reward(
                    flatten_states, flatten_rnn_hiddens
                ).view(self.chunk_length - 1, self.batch_size, 1)

                # 観測と報酬の予測誤差を計算
                obs_loss = (
                    0.5
                    * F.mse_loss(recon_observations, observations[1:], reduction="none")
                    .mean([0, 1])
                    .sum()
                )
                reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

                # 以上のロスを合わせて勾配降下で更新する
                model_loss = kl_loss + obs_loss + reward_loss
                self.model_optimizer.zero_grad()
                model_loss.backward()
                clip_grad_norm_(self.model_params, self.clip_grad_norm)
                self.model_optimizer.step()

                # --------------------------------------------------
                #  Action Model, Value　Modelの更新　- Behavior leaning
                # --------------------------------------------------
                # Actor-Criticのロスで他のモデルを更新することはないので勾配の流れを一度遮断
                # flatten_states, flatten_rnn_hiddensは RSSMから得られた低次元の状態表現を平坦化した値
                flatten_states = flatten_states.detach()
                flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

                # DreamerにおけるActor-Criticの更新のために, 現在のモデルを用いた
                # 数ステップ先の未来の状態予測を保持するためのTensorを用意
                imaginated_states = torch.zeros(
                    self.imagination_horizon + 1,
                    *flatten_states.shape,
                    device=flatten_states.device
                )
                imaginated_rnn_hiddens = torch.zeros(
                    self.imagination_horizon + 1,
                    *flatten_rnn_hiddens.shape,
                    device=flatten_rnn_hiddens.device
                )

                # 　未来予測をして想像上の軌道を作る前に, 最初の状態としては先ほどモデルの更新で使っていた
                # リプレイバッファからサンプルされた観測データを取り込んだ上で推論した状態表現を使う
                imaginated_states[0] = flatten_states
                imaginated_rnn_hiddens[0] = flatten_rnn_hiddens

                # open-loopで未来の状態予測を使い, 想像上の軌道を作る
                for h in range(1, self.imagination_horizon + 1):
                    # 行動はActionModelで決定. この行動はモデルのパラメータに対して微分可能で,
                    # 　これを介してActionModelは更新される
                    actions = self.action_model(flatten_states, flatten_rnn_hiddens)
                    (
                        flatten_states_prior,
                        flatten_rnn_hiddens,
                    ) = self.rssm.transition.prior(
                        self.rssm.transition.reccurent(
                            flatten_states, actions, flatten_rnn_hiddens
                        )
                    )
                    flatten_states = flatten_states_prior.rsample()
                    imaginated_states[h] = flatten_states
                    imaginated_rnn_hiddens[h] = flatten_rnn_hiddens

                # RSSMのreward_modelにより予測された架空の軌道に対する報酬を計算
                flatten_imaginated_states = imaginated_states.view(-1, self.state_dim)
                flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(
                    -1, self.rnn_hidden_dim
                )
                imaginated_rewards = self.rssm.reward(
                    flatten_imaginated_states, flatten_imaginated_rnn_hiddens
                ).view(self.imagination_horizon + 1, -1)
                imaginated_values = self.value_model(
                    flatten_imaginated_states, flatten_imaginated_rnn_hiddens
                ).view(self.imagination_horizon + 1, -1)

                # λ-returnのターゲットを計算(V_{\lambda}(s_{\tau})
                lambda_target_values = lambda_target(
                    imaginated_rewards, imaginated_values, self.gamma, self.lambda_
                )

                # 価値関数の予測した価値が大きくなるようにActionModelを更新
                # PyTorchの基本は勾配降下だが, 今回は大きくしたいので-1をかける
                action_loss = -lambda_target_values.mean()
                self.action_optimizer.zero_grad()
                action_loss.backward()
                clip_grad_norm_(self.action_model.parameters(), self.clip_grad_norm)
                self.action_optimizer.step()

                # TD(λ)ベースの目的関数で価値関数を更新（価値関数のみを学習するため，学習しない変数のグラフは切っている. )
                imaginated_values = self.value_model(
                    flatten_imaginated_states.detach(),
                    flatten_imaginated_rnn_hiddens.detach(),
                ).view(self.imagination_horizon + 1, -1)
                value_loss = 0.5 * F.mse_loss(
                    imaginated_values, lambda_target_values.detach()
                )
                self.value_optimizer.zero_grad()
                value_loss.backward()
                clip_grad_norm_(self.value_model.parameters(), self.clip_grad_norm)
                self.value_optimizer.step()

                # ログをTensorBoardに出力
                print(
                    "update_step: %3d model loss: %.5f, kl_loss: %.5f, "
                    "obs_loss: %.5f, reward_loss: %.5f, "
                    "value_loss: %.5f action_loss: %.5f"
                    % (
                        update_step + 1,
                        model_loss.item(),
                        kl_loss.item(),
                        obs_loss.item(),
                        reward_loss.item(),
                        value_loss.item(),
                        action_loss.item(),
                    )
                )
                total_update_step = episode * self.collect_interval + update_step
                self.writer.add_scalar(
                    "model loss", model_loss.item(), total_update_step
                )
                self.writer.add_scalar("kl loss", kl_loss.item(), total_update_step)
                self.writer.add_scalar("obs loss", obs_loss.item(), total_update_step)
                self.writer.add_scalar(
                    "reward loss", reward_loss.item(), total_update_step
                )
                self.writer.add_scalar(
                    "value loss", value_loss.item(), total_update_step
                )
                self.writer.add_scalar(
                    "action loss", action_loss.item(), total_update_step
                )

            print("elasped time for update: %.2fs" % (time.time() - start))

            # --------------------------------------------------------------
            #    テストフェーズ. 探索ノイズなしでの性能を評価する
            # --------------------------------------------------------------
            if (episode + 1) % self.test_interval == 0:
                policy = Agent(self.encoder, self.rssm.transition, self.action_model)
                start = time.time()
                obs = self.env.reset()
                done = False
                total_reward = 0
                while not done:
                    action = policy(obs, training=False)
                    obs, reward, done, _ = self.env.step(action)
                    total_reward += reward

                self.writer.add_scalar("total reward at test", total_reward, episode)
                print(
                    "Total test reward at episode [%4d/%4d] is %f"
                    % (episode + 1, self.all_episodes, total_reward)
                )
                print("elasped time for test: %.2fs" % (time.time() - start))

            if (episode + 1) % self.model_save_interval == 0:
                # 定期的に学習済みモデルのパラメータを保存する
                model_log_dir = os.path.join(
                    self.model_save_dir, "episode_%04d" % (episode + 1)
                )
                os.makedirs(model_log_dir)
                torch.save(
                    self.encoder.state_dict(),
                    os.path.join(model_log_dir, "encoder.pth"),
                )
                torch.save(
                    self.rssm.transition.state_dict(),
                    os.path.join(model_log_dir, "rssm.pth"),
                )
                torch.save(
                    self.rssm.observation.state_dict(),
                    os.path.join(model_log_dir, "obs_model.pth"),
                )
                torch.save(
                    self.rssm.reward.state_dict(),
                    os.path.join(model_log_dir, "reward_model.pth"),
                )
                torch.save(
                    self.value_model.state_dict(),
                    os.path.join(model_log_dir, "value_model.pth"),
                )
                torch.save(
                    self.action_model.state_dict(),
                    os.path.join(model_log_dir, "action_model.pth"),
                )

        self.writer.close()

    def load_models(self, model_dir):
        self.encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pth")))
        self.rssm.transition.load_state_dict(
            torch.load(os.path.join(model_dir, "rssm.pth"))
        )
        self.rssm.observation.load_state_dict(
            torch.load(os.path.join(model_dir, "obs_model.pth"))
        )
        self.action_model.load_state_dict(
            torch.load(os.path.join(model_dir, "action_model.pth"))
        )

    def view(self, test_count):
        policy = Agent(self.encoder, self.rssm.transition, self.action_model)

        for i in range(test_count):
            obs = self.env.reset()
            done = False
            total_reward = 0
            frames = [Image.fromarray(obs)]

            while not done:
                action = policy(obs, training=False)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                frames.append(Image.fromarray(obs))

            print("Total Reward:", total_reward)
            frames[0].save(
                os.path.join(self.gif_dir, "test" + str(i) + ".gif"),
                save_all=True,
                append_images=frames[1:],
                duration=40,
            )
