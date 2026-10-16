"""Attribute DMA traffic / TensorE work to tensors and HLO ops for an ingested profile.

usage: python attrib.py <display-name>
"""
import os
import sys

import duckdb

WORK = os.environ.get("ASR_WORK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(WORK, "profiles", "ne-data", "profiles", "global", f"{sys.argv[1]}@latest") + "/"
c = duckdb.connect()
for t in ["DmaPacket", "Instruction", "TensorInfo", "FrameworkNode"]:
    c.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{P}{t}.parquet')")


def show(title, sql):
    print(f"\n-- {title}")
    print(c.execute(sql).df().to_string(index=False, max_colwidth=70))


show("DMA by variable (top 15 by bytes)", """
SELECT variable, COUNT(*) pkts, ROUND(SUM(transfer_bytes)/1e6,1) MB, ROUND(AVG(transfer_bytes)) avgB,
       ROUND(SUM(duration_ns)/1e6,1) pkt_ms
FROM DmaPacket WHERE queue_type != 'instruction' AND transfer_bytes > 4
GROUP BY 1 ORDER BY MB DESC LIMIT 15""")

show("TensorInfo load-to-SBUF (top 14 by total loaded)", """
SELECT variable_name, shape, ROUND(size/1e6,2) MB, load_to_sbuf_dma_count dmas,
       ROUND(load_to_sbuf_avg_size_bytes) avgB, ROUND(load_to_sbuf_repeat_factor,2) rep,
       ROUND(load_to_sbuf_total_size_bytes/1e6,1) loadedMB
FROM TensorInfo WHERE type IN ('IN','OUT')
ORDER BY load_to_sbuf_total_size_bytes DESC NULLS LAST LIMIT 14""")

show("TensorE MATMUL by instruction type", """
SELECT tensor_instruction_type, COUNT(*) n, ROUND(SUM(duration_ns)/1e6,2) ms, ROUND(SUM(adjusted_flops)/1e9,1) GF
FROM Instruction WHERE engine='Tensor' AND opcode='MATMUL' GROUP BY 1 ORDER BY ms DESC""")

show("Instruction time by HLO op (top 20)", """
SELECT regexp_replace(hlo_name, '[.][0-9]+$', '') hlo, COUNT(*) n,
       ROUND(SUM(duration_ns)/1e6,2) ms,
       ROUND(SUM(CASE WHEN engine='Tensor' THEN duration_ns END)/1e6,2) te_ms,
       ROUND(SUM(CASE WHEN opcode='DMA_DIRECT2D' THEN duration_ns END)/1e6,2) swdma_ms,
       ROUND(SUM(hbm_read_bytes)/1e6,1) hbmR_MB, ROUND(SUM(hbm_write_bytes)/1e6,1) hbmW_MB
FROM Instruction GROUP BY 1 ORDER BY ms DESC LIMIT 20""")

show("Framework nodes (top 20 by HBM bytes)", """
SELECT node_name, ROUND((end_ts-start_ts)/1e6,2) span_ms, ROUND((hbm_read_bytes+hbm_write_bytes)/1e6,1) hbm_MB,
       ROUND(TRY_CAST(total_time_tensor AS DOUBLE)/1e6,2) te_ms, ROUND(TRY_CAST(total_time_gpsimd AS DOUBLE)/1e6,2) pool_ms,
       ROUND(TRY_CAST(total_time_scalar AS DOUBLE)/1e6,2) act_ms
FROM FrameworkNode ORDER BY hbm_read_bytes+hbm_write_bytes DESC NULLS LAST LIMIT 20""")
