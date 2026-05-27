cd /root/proj/openpi
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main_o_spatial.py  > /root/proj/eval/openpi/spatial.log 2>&1
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main_o_object.py  > /root/proj/eval/openpi/object.log 2>&1
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main_o_goal.py  > /root/proj/eval/openpi/goal.log 2>&1
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main_o_10.py  > /root/proj/eval/openpi/10.log 2>&1
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main_o_90.py  > /root/proj/eval/openpi/90.log 2>&1