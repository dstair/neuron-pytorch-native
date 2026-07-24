import argparse
import ctypes


FAMILIES = {
    "trn1": 2,
    "trn2": 5,
}


class InstanceInfo(ctypes.Structure):
    _fields_ = [
        ("family", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("arch_name", ctypes.c_char * 16),
        ("device_revision", ctypes.c_char * 8),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", choices=FAMILIES, required=True)
    parser.add_argument("--physical", choices=FAMILIES, required=True)
    args = parser.parse_args()

    from torch_neuronx import _C

    reported_target = _C._get_platform_target()
    if reported_target != args.physical:
        raise RuntimeError(
            f"Torch NeuronX reported {reported_target}, expected physical {args.physical}"
        )

    # Resolve through the process-global scope so this call traverses LD_PRELOAD.
    nrt = ctypes.CDLL(None)
    nrt.nrt_get_instance_info.argtypes = [
        ctypes.POINTER(InstanceInfo),
        ctypes.c_size_t,
    ]
    nrt.nrt_get_instance_info.restype = ctypes.c_int
    info = InstanceInfo()
    status = nrt.nrt_get_instance_info(ctypes.byref(info), ctypes.sizeof(info))
    if status != 0:
        raise RuntimeError(f"direct nrt_get_instance_info failed with status {status}")
    if info.family != FAMILIES[args.physical]:
        raise RuntimeError(
            f"direct NRT family was {info.family}, expected {FAMILIES[args.physical]}"
        )

    print(f"torch_neuronx_target={reported_target}")
    print(f"direct_nrt_family={info.family}")
    print(f"cache_platform_override={args.override}")


if __name__ == "__main__":
    main()
