import os

from torch_neuronx import _C


print(
    f"pid={os.getpid()} preload={os.environ.get('LD_PRELOAD')} "
    f"cache_target={os.environ.get('QWEN35_CACHE_PLATFORM_TARGET')}",
    flush=True,
)
try:
    _C.compile_graph("qwen35-cache-override-validation", b"invalid", False)
except RuntimeError:
    pass
