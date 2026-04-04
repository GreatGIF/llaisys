set -e
# xmake f --root --openmp=y --cpu-blas=n --cpu-mkl=n --cpu-onednn=n --nv-gpu=n --mx-gpu=y -c -v
cd /root/code/llaisys && xmake f --root --openmp=y --cpu-blas=n --cpu-mkl=n --cpu-onednn=n --nv-gpu=n --mx-gpu=y -c -v # 沐曦环境
xmake --root
xmake --root install

find build -type f -name 'lib*.so' -path '*/release/*' -exec cp -f {} python/llaisys/libllaisys/ \; # 沐曦环境

# export LLAISYS_LINEAR_LOG_BACKEND=1 # 启用linear采用的后端log
# export LLAISYS_OP_PROFILE=1 # 启用server在batch infer时的ops profile信息

pip install -e ./python/
# pip install ./python/

# script_name="test/$1.py"

# if [ ! -f "$script_name" ]; then
#     echo "错误：脚本文件不存在: $script_name" >&2
#     exit 1
# fi

# python "$script_name"

# python test/test_infer.py --model /mnt/d/model/Qwen2.5-1.5B/ --test

# xmake f -c --nv-gpu=y
# xmake
# xmake install
# ./build/linux/x86_64/release/scheduler-test
# ./build/linux/x86_64/release/qwen2-paged-test
# ./build/linux/x86_64/release/dynamic-batch-test
# ./build/linux/x86_64/release/qwen2-c-api-test
# ./build/linux/x86_64/release/qwen2-paged-test
# ./build/linux/x86_64/release/dynamic-batch-test