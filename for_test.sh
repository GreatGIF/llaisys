set -e
xmake f -cv --openmp=y --cpu-blas=y --cpu-mkl=y --cpu-onednn=y --nv-gpu=y
xmake
xmake install
export LLAISYS_LINEAR_LOG_BACKEND=1
# export LLAISYS_LINEAR_FORCE_BACKEND=onednn

# pip install -e ./python/
# pip install ./python/

# script_name="test/$1.py"

# if [ ! -f "$script_name" ]; then
#     echo "错误：脚本文件不存在: $script_name" >&2
#     exit 1
# fi

# python "$script_name"

# python test/test_infer.py --model /mnt/d/model/Qwen2.5-1.5B/ --test