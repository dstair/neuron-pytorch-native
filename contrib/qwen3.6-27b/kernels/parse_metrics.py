"""Parse a neuron-explorer summary-json metrics file and print human-readable summary."""
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
d = list(data.values())[0]

keys = [
    ('total_time',              'Total kernel wall-clock', 'us'),
    ('total_active_time',       'Total active time',       'us'),
    ('total_active_time_percent','Active time',            '%'),
    ('tensor_engine_active_time', 'TensorE active',         'us'),
    ('tensor_engine_active_time_percent', 'TensorE',          '%'),
    ('vector_engine_active_time', 'VectorE active',         'us'),
    ('vector_engine_active_time_percent', 'VectorE',          '%'),
    ('scalar_engine_active_time', 'ScalarE active',         'us'),
    ('scalar_engine_active_time_percent', 'ScalarE',          '%'),
    ('gpsimd_engine_active_time', 'GpSimdE active',         'us'),
    ('gpsimd_engine_active_time_percent', 'GpSimdE',          '%'),
    ('dma_active_time',          'DMA active',             'us'),
    ('dma_active_time_percent',  'DMA',                    '%'),
    ('hbm_read_bytes',           'HBM read',               'bytes'),
    ('hbm_write_bytes',          'HBM write',              'bytes'),
    ('mfu_estimated_percent',    'MFU est',                '%'),
    ('mbu_estimated_percent',    'MBU est',                '%'),
    ('mm_arithmetic_intensity',  'MM arith intensity',     ''),
    ('peak_flops_bandwidth_ratio','Peak FLOPS/BW',         ''),
    ('matmul_instruction_count', 'MatMul instr',           ''),
    ('activate_instruction_count','Activation instr',      ''),
    ('vector_engine_instruction_count', 'VectorE instr',    ''),
    ('scalar_engine_instruction_count', 'ScalarE instr',    ''),
    ('gpsimd_engine_instruction_count', 'GpSimdE instr',    ''),
    ('dma_transfer_count',       'DMA transfers',          ''),
    ('dma_transfer_total_bytes', 'DMA transfer total',     'bytes'),
    ('software_dynamic_dma_packet_count', 'Soft dyn DMA pkts', ''),
    ('spill_reload_bytes',       'Spill reload',           'bytes'),
    ('spill_save_bytes',         'Spill save',             'bytes'),
]

for k, label, unit in keys:
    v = d.get(k)
    if v is None:
        continue
    if unit == 'us' and isinstance(v, (int, float)):
        print(f"  {label:30s}  {v*1e6:10.2f} us")
    elif unit == '%' and isinstance(v, (int, float)):
        print(f"  {label:30s}  {v*100:10.2f} %")
    elif unit == 'bytes' and isinstance(v, (int, float)):
        print(f"  {label:30s}  {v:>10,} bytes")
    elif isinstance(v, float):
        print(f"  {label:30s}  {v:>10.4g}")
    else:
        print(f"  {label:30s}  {v}")
