import dataclasses


@dataclasses.dataclass
class DreamerConfig:
    buffer_capacity: int = 500000  # colabのメモリに合わせて変えたほうがいいかも

    # state_dim
    state_dim: int = 30
    rnn_hidden_dim: int = 200

    # learning rate and epsilon
    model_lr: float = 6e-4
    value_lr: float = 8e-5
    action_lr: float = 8e-5
    eps: float = 1e-4

    # other hyper params
    seed_episodes: int = 5
    all_episodes: int = 300
    test_interval: int = 10
    model_save_interval: int = 100
    collect_interval: int = 50

    action_noise_var: float = 0.3

    batch_size: int = 50
    chunk_length: int = 50
    imagination_horizon: int = 15

    gamma: float = 0.99
    lambda_: float = 0.95
    clip_grad_norm: float = 100.0
    free_nats: float = 3.0
