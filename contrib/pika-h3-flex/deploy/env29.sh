# Source for the minimax-h3-single-device (0.29) stack on the trn2.3xl.
export OMNI=/data/omni29
export PATH=$OMNI/venv/bin:/opt/aws/neuron/bin:$PATH
export LD_LIBRARY_PATH=/opt/aws/neuron/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VLLM_NEURON_BACKEND=neuron_native
export VLLM_NEURON_LIBTORCH_NEURONX_LITE=1
export TORCH_CACHING_PRECOMPILE=0
export HF_HOME=$OMNI/huggingface
export VLLM_CACHE_ROOT=$OMNI/vllm-cache
export NKI_COMPILE_CACHE_URL=$OMNI/nki-cache
export TMPDIR=$OMNI/tmp
# The tests resolve MiniMaxAI/MiniMax-H3 under $FSX_SHARED_RO/neuronx-distributed/inference/.
export FSX_SHARED_RO=$OMNI/fsx
mkdir -p $TMPDIR $HF_HOME $VLLM_CACHE_ROOT $NKI_COMPILE_CACHE_URL $OMNI/fsx/neuronx-distributed/inference/MiniMaxAI
[ -e $OMNI/fsx/neuronx-distributed/inference/MiniMaxAI/MiniMax-H3 ] || \
  ln -s /data/models/MiniMax-H3 $OMNI/fsx/neuronx-distributed/inference/MiniMaxAI/MiniMax-H3
