set -e
xmake f -c --openmp=y --cpu-blas=y
xmake
xmake install
# pip install -e ./python/
# pip install ./python/

# script_name="test/$1.py"

# if [ ! -f "$script_name" ]; then
#     echo "错误：脚本文件不存在: $script_name" >&2
#     exit 1
# fi

# python "$script_name"

# python test/test_infer.py --model /mnt/d/model/Qwen2.5-1.5B/ --test