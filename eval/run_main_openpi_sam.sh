cd openpi
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main_goal.py --args.use-sam > /root/proj/goal_sam.log 2>&1
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main_10.py --args.use-sam > /root/proj/10_sam.log 2>&1
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main_90.py --args.use-sam > /root/proj/90_sam.log 2>&1