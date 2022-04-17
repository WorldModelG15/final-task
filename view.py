from utils.env import launch_env
from utils.wrappers import (
    OriginalWrapper,
    NormalizeWrapper,
    ImgWrapper,
    DtRewardWrapper,
    ActionWrapper,
    ResizeWrapper,
)
from dreamer.config import DreamerConfig
import torch
from dreamer.trainer import Trainer

if __name__ == "__main__":
    env = launch_env(map_name="loop_pedestrians")
    env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 120x160x3 into 3x120x160
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = OriginalWrapper(env)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = DreamerConfig()
    trainer = Trainer(env, device, config, False)
    trainer.load_models("/root/mnt/final-task/models/20220416132240/episode_0300")

    trainer.view(10)
