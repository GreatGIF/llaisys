#!/usr/bin/env bash
set -euo pipefail

# 解析真实 mxcc 路径
mxcc_bin="${MXCC_REAL:-}"
if [[ -z "${mxcc_bin}" ]]; then
  if command -v mxcc >/dev/null 2>&1; then
    mxcc_bin="$(command -v mxcc)"
  else
    for p in \
      /opt/maca/mxgpu_llvm/bin/mxcc \
      /usr/local/maca/mxgpu_llvm/bin/mxcc \
      /opt/mx/mxgpu_llvm/bin/mxcc \
      /usr/local/mx/mxgpu_llvm/bin/mxcc \
      /opt/maca/bin/mxcc \
      /usr/local/maca/bin/mxcc \
      /opt/mx/bin/mxcc \
      /usr/local/mx/bin/mxcc
    do
      if [[ -x "$p" ]]; then
        mxcc_bin="$p"
        break
      fi
    done
  fi
fi

if [[ -z "${mxcc_bin}" || ! -x "${mxcc_bin}" ]]; then
  echo "mxcc wrapper error: mxcc not found. Set MXCC_REAL or add mxcc to PATH." >&2
  exit 127
fi

# 过滤/重写参数：
# 1) 去掉 mxcc 不支持或无意义参数
# 2) 去掉 xmake 对 sourcekind=cxx 注入的 "-x c++"，避免覆盖 "-x maca"
filtered=()
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
  arg="${args[$i]}"

  case "$arg" in
    -m64|-m32|-Qunused-arguments|-finput-charset=UTF-8|-fexec-charset=UTF-8|-s)
      ((i+=1))
      continue
      ;;
    -x)
      next=""
      if [[ $((i+1)) -lt ${#args[@]} ]]; then
        next="${args[$((i+1))]}"
      fi
      if [[ "$next" == "c++" || "$next" == "c" ]]; then
        # 丢弃 "-x c++" / "-x c" 这对参数
        ((i+=2))
        continue
      fi
      ;;
  esac

  filtered+=("$arg")
  ((i+=1))
done

exec "${mxcc_bin}" "${filtered[@]}"
