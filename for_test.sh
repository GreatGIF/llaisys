set -e
cd /root/code/llaisys && xmake f --root --openmp=y --cpu-blas=n --cpu-mkl=n --cpu-onednn=n --nv-gpu=n --mx-gpu=y -c -v
xmake --root
xmake --root install

# 兜底：当前产物在 build/**/release，确保 Python 包目录拿到所有 .so
find build -type f -name 'lib*.so' -path '*/release/*' -exec cp -f {} python/llaisys/libllaisys/ \;

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