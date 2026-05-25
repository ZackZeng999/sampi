import collections
import dataclasses
import logging
import math
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from sam_dim_client import SamDimClient
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # Optional SAM pre-processing
    #################################################################################################################
    use_sam: bool = False
    extract_url: str = ""  # Optional VLM extractor endpoint. Empty means use the local placeholder.
    sam_url: str = "http://127.0.0.1:9001/segment"
    sam_prompt: str = ""  # Optional comma-separated object prompts, e.g. "black bowl, plate".
    sam_view: str = "base"  # Options: base, wrist, both
    sam_background_scale: float = 0.4
    sam_score_threshold: float = 0.6
    sam_max_masks_per_prompt: int = 3
    sam_blur_radius: float = 1.5
    sam_timeout_sec: float = 600.0
    sam_fail_open: bool = True

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    sam_client = None
    sam_views = set()
    if args.use_sam:
        sam_client = SamDimClient(
            sam_url=args.sam_url,
            extract_url=args.extract_url,
            prompt=args.sam_prompt or None,
            background_scale=args.sam_background_scale,
            score_threshold=args.sam_score_threshold,
            max_masks_per_prompt=args.sam_max_masks_per_prompt,
            blur_radius=args.sam_blur_radius,
            timeout_sec=args.sam_timeout_sec,
            fail_open=args.sam_fail_open,
        )
        sam_views = {"base", "wrist"} if args.sam_view == "both" else {args.sam_view}
        invalid_views = sam_views - {"base", "wrist"}
        if invalid_views:
            raise ValueError(f"Unknown sam_view value: {args.sam_view}. Use base, wrist, or both.")
        logging.info("Using SAM dim-background preprocessing for views: %s", sorted(sam_views))


    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        cached_task_prompts = None

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_img = img
                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        policy_img = img
                        policy_wrist_img = wrist_img
                        if sam_client is not None and cached_task_prompts is None:
                            extract_start = time.perf_counter()
                            cached_task_prompts = sam_client.extract_prompts_for_image(str(task_description), img)
                            extract_duration = time.perf_counter() - extract_start
                            logging.info(
                                "SAM extract took %.3fs for task %r and returned prompts=%s",
                                extract_duration,
                                task_description,
                                cached_task_prompts,
                            )
                        if sam_client is not None and cached_task_prompts:
                            sam_segment_start = time.perf_counter()
                            if "base" in sam_views:
                                policy_img = sam_client.dim_background_with_prompts(policy_img, cached_task_prompts)
                            if "wrist" in sam_views:
                                policy_wrist_img = sam_client.dim_background_with_prompts(policy_wrist_img, cached_task_prompts)
                            sam_segment_duration = time.perf_counter() - sam_segment_start
                            logging.info(
                                "SAM segment/dim took %.3fs for views=%s using prompts=%s",
                                sam_segment_duration,
                                sorted(sam_views),
                                cached_task_prompts,
                            )
                        replay_img = policy_img

                        # Prepare observations dict
                        element = {
                            "observation/image": policy_img,
                            "observation/wrist_image": policy_wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        openpi_infer_start = time.perf_counter()
                        action_chunk = client.infer(element)["actions"]
                        openpi_infer_duration = time.perf_counter() - openpi_infer_start
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        planned_actions = action_chunk[: args.replan_steps]
                        action_plan.extend(planned_actions)
                        logging.info(
                            "OpenPI infer took %.3fs and produced %d actions (using first %d for this chunk)",
                            openpi_infer_duration,
                            len(action_chunk),
                            len(planned_actions),
                        )

                    replay_images.append(replay_img)
                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
