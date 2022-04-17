import gym
import gym_duckietown


def launch_env(id=None, map_name="loop_empty", is_original_map=False, map_abs_path=""):
    env = None
    if id is None:
        # Launch the environment
        from gym_duckietown.simulator import Simulator

        env = Simulator(
            seed=123,  # random seed
            map_name=map_name,
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=True,
            is_original_map=is_original_map,
            map_abs_path = map_abs_path,
        )
    else:
        env = gym.make(id)

    return env
