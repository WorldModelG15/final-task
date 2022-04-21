from utils.env import launch_env
from utils.wrappers import (
    OriginalWrapper,
    ImgWrapper,
    DtRewardWrapper,
    ActionWrapper,
    ResizeWrapper,
)
from dreamer.config import DreamerConfig
import torch
from dreamer.trainer import Trainer

if __name__ == "__main__":
    ### デフォルトのマップで学習する場合
    env = launch_env(map_name="loop_pedestrians")

    ### オリジナルのマップで学習する場合
    # map_dir_abs_path = (
    #     "/root/mnt/final-task/gym-duckietown/created_maps/"  # ここは環境によって変えます
    # )
    # map_name = "zigzag"  # 'zigzag','oneloop','three_statics','loop_pedestrian'から選択です
    # env = launch_env(
    #     is_original_map=True, map_abs_path=map_dir_abs_path + map_name + ".yaml"
    # )

    env = ResizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 120x160x3 into 3x120x160
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = OriginalWrapper(env)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = DreamerConfig()
    trainer = Trainer(env, device, config, False)
    trainer.load_models("/root/mnt/final-task/models/20220420030645/episode_0300")

    trainer.compare_imagination(10)
