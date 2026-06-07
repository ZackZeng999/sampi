unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0
cd /root/proj/openpi
conda activate openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python /root/proj/openpi/examples/libero/main_10.py --args.use-sam > /root/proj/eval/openpi_finetune2epoch/10.log 2>&1
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python /root/proj/openpi/examples/libero/main_spatial.py --args.use-sam > /root/proj/eval/openpi_finetune2epoch/spatial.log 2>&1
