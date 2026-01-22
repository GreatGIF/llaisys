set -e
xmake
xmake install
pip install ./python/

script_name="test/$1.py"

if [ ! -f "$script_name" ]; then
    echo "错误：脚本文件不存在: $script_name" >&2
    exit 1
fi

python "$script_name"
