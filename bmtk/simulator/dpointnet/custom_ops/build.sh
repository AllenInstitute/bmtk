#!/usr/bin/env bash
set -euo pipefail

custom_ops_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python}"
prefix="$("$python" -c 'import sys; print(sys.prefix)')"
nvcc="${NVCC:-$prefix/bin/nvcc}"
cxx="${CXX:-$prefix/bin/x86_64-conda-linux-gnu-g++}"
build_dir="${DPOINTNET_CUSTOM_OP_BUILD_DIR:-$custom_ops_dir/build}"
output="${DPOINTNET_CUSTOM_OP_OUTPUT:-$custom_ops_dir/_csr_spike_ops.so}"

if [[ ! -x "$nvcc" ]]; then
  echo "nvcc was not found at $nvcc" >&2
  exit 1
fi
if [[ ! -x "$cxx" ]]; then
  cxx="${CXX:-c++}"
fi

mapfile -t tf_compile_flags < <(
  "$python" -c 'import tensorflow as tf; print(*tf.sysconfig.get_compile_flags(), sep="\n")'
)
mapfile -t tf_link_flags < <(
  "$python" -c 'import tensorflow as tf; print(*tf.sysconfig.get_link_flags(), sep="\n")'
)

read -r -a cuda_archs <<<"${DPOINTNET_CUDA_ARCHS:-70 75 80 86 89 90}"

gencode_flags=()
for arch in "${cuda_archs[@]}"; do
  gencode_flags+=("-gencode=arch=compute_${arch},code=sm_${arch}")
done
highest_arch="${cuda_archs[${#cuda_archs[@]}-1]}"
gencode_flags+=(
  "-gencode=arch=compute_${highest_arch},code=compute_${highest_arch}"
)

mkdir -p "$build_dir"
"$cxx" -std=c++17 -fPIC -O3 \
  -I"$prefix/include" \
  "${tf_compile_flags[@]}" \
  -c "$custom_ops_dir/csr_spike_ops.cc" \
  -o "$build_dir/csr_spike_ops.o"

"$nvcc" -ccbin "$cxx" -std=c++17 -x cu -Xcompiler=-fPIC -O3 \
  --expt-relaxed-constexpr \
  -DGOOGLE_CUDA=1 \
  -I"$prefix/include" \
  "${tf_compile_flags[@]}" \
  "${gencode_flags[@]}" \
  -c "$custom_ops_dir/csr_spike_ops.cu.cc" \
  -o "$build_dir/csr_spike_ops.cu.o"

"$cxx" -shared \
  "$build_dir/csr_spike_ops.o" \
  "$build_dir/csr_spike_ops.cu.o" \
  "${tf_link_flags[@]}" \
  -L"$prefix/lib" -lcudart \
  -Wl,-rpath,"$prefix/lib" \
  -o "$output"

arch_file="${output%.so}.archs"
{
  printf 'sm=%s\n' "${cuda_archs[*]}"
  printf 'ptx=%s\n' "$highest_arch"
} > "$arch_file"

echo "$output"
