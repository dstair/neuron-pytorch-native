"""Module / source-line / DMA-queue / timeline breakdown for an ingested decoder-step profile.

usage: python deep.py <display-name>
Relies on op-level attribution (Instruction.layer = "<Module>[<name>][<idx>]_<hlo>", needs debug-info compile).
"""
import os
import sys

import duckdb

WORK = os.environ.get("ASR_WORK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(WORK, "profiles", "ne-data", "profiles", "global", f"{sys.argv[1]}@latest") + "/"
c = duckdb.connect()
for t in ["Instruction", "DmaPacket", "ActiveTime", "TensorInfo"]:
    c.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{P}{t}.parquet')")
c.execute("""CREATE VIEW I AS SELECT *,
  coalesce(nullif(regexp_extract(layer, '^([A-Za-z]+\\[[a-z0-9]+\\])', 1), ''), 'other:' || coalesce(nullif(regexp_extract(layer, '_([a-z]+)\\.[0-9]+$', 1), ''), '?')) AS mod,
  try_cast(nullif(regexp_extract(layer, '\\]\\[([0-9]+)\\]', 1), '') AS INT) AS lidx
  FROM Instruction""")


def show(title, sql):
    print(f"\n-- {title}")
    print(c.execute(sql).df().to_string(index=False, max_colwidth=60))


t0, t1 = c.execute("SELECT MIN(start_ts), MAX(end_ts) FROM Instruction").fetchone()
print(f"window {(t1-t0)/1e6:.3f} ms")

show("busy time by module x engine (summed instr ms)", """
SELECT mod, COUNT(*) n,
  ROUND(SUM(CASE WHEN engine='Tensor' THEN duration_ns END)/1e6,2) pe,
  ROUND(SUM(CASE WHEN engine='Vector' THEN duration_ns END)/1e6,2) dve,
  ROUND(SUM(CASE WHEN engine='Scalar' THEN duration_ns END)/1e6,2) act,
  ROUND(SUM(CASE WHEN engine='GpSimd' THEN duration_ns END)/1e6,2) pool,
  ROUND(SUM(CASE WHEN opcode='DMA_DIRECT2D' THEN duration_ns END)/1e6,2) sw_dma,
  ROUND(SUM(hbm_read_bytes)/1e6,1) hbmR, ROUND(SUM(hbm_write_bytes)/1e6,1) hbmW
FROM I GROUP BY 1 ORDER BY COALESCE(pe,0)+COALESCE(dve,0)+COALESCE(act,0)+COALESCE(pool,0) DESC LIMIT 18""")

show("source line hotspots (top 15)", """
SELECT regexp_replace(bir_debug_info_source_location, '.*/', '') src, engine, COUNT(*) n, ROUND(SUM(duration_ns)/1e6,2) ms
FROM I WHERE bir_debug_info_source_location IS NOT NULL GROUP BY 1,2 ORDER BY ms DESC LIMIT 15""")

show("DMA by queue class", """
SELECT regexp_replace(queue_name, '-E[0-9]+$', '') q, queue_type, COUNT(*) pkts, ROUND(SUM(transfer_bytes)/1e6,1) MB,
  ROUND(AVG(transfer_bytes)) avgB, ROUND(SUM(transfer_bytes)/NULLIF(SUM(duration_ns),0),2) GBps_per_pkt
FROM DmaPacket WHERE transfer_bytes > 4 GROUP BY 1,2 ORDER BY MB DESC LIMIT 12""")

show("DMA packet size histogram (bytes-weighted)", """
SELECT CASE WHEN transfer_bytes<1024 THEN 'a <1KB' WHEN transfer_bytes<2048 THEN 'b 1-2KB' WHEN transfer_bytes<4096 THEN 'c 2-4KB'
  WHEN transfer_bytes<16384 THEN 'd 4-16KB' ELSE 'e >=16KB' END bucket, COUNT(*) pkts, ROUND(SUM(transfer_bytes)/1e6,1) MB,
  ROUND(SUM(duration_ns)/1e6,1) pkt_ms, ROUND(SUM(transfer_bytes)/NULLIF(SUM(duration_ns),0),2) GBps_per_pkt
FROM DmaPacket WHERE queue_type != 'instruction' AND transfer_bytes > 4 GROUP BY 1 ORDER BY 1""")

show("weight tensors: DMA packet sizes (top 10 by MB)", """
SELECT d.variable, t.shape, COUNT(*) pkts, ROUND(SUM(d.transfer_bytes)/1e6,1) MB, ROUND(AVG(d.transfer_bytes)) avgB
FROM DmaPacket d LEFT JOIN TensorInfo t ON t.variable_name = d.variable
WHERE d.transfer_bytes > 4 GROUP BY 1,2 ORDER BY MB DESC LIMIT 10""")

show("per decoder layer wall span + PE busy (first/last 3 and stats)", """
WITH L AS (SELECT lidx, MIN(start_ts) s, MAX(end_ts) e,
  SUM(CASE WHEN engine='Tensor' THEN duration_ns END) pe FROM I WHERE lidx IS NOT NULL GROUP BY 1)
SELECT COUNT(*) layers, ROUND(AVG((e-s)/1e6),3) avg_span_ms, ROUND(MIN((e-s)/1e6),3) min_span, ROUND(MAX((e-s)/1e6),3) max_span,
  ROUND(AVG(pe/1e6),3) avg_pe_ms, ROUND((MAX(e)-MIN(s))/1e6,3) layers_window_ms FROM L""")

show("engine idle: largest gaps in DMA activity (top 8)", """
WITH p AS (SELECT start_ts s, end_ts e FROM DmaPacket WHERE transfer_bytes > 4),
o AS (SELECT s, e, MAX(e) OVER (ORDER BY s ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_e FROM p)
SELECT ROUND((s-prev_e)/1e3,1) gap_us, ROUND((prev_e-(SELECT MIN(start_ts) FROM Instruction))/1e6,3) at_ms
FROM o WHERE s > prev_e ORDER BY s-prev_e DESC LIMIT 8""")

show("what runs at the very end (last 0.5 ms) by module", f"""
SELECT mod, engine, COUNT(*) n, ROUND(SUM(duration_ns)/1e6,3) ms FROM I WHERE start_ts > {t1} - 500000
GROUP BY 1,2 ORDER BY ms DESC LIMIT 8""")
show("what runs at the very start (first 0.5 ms) by module", f"""
SELECT mod, engine, COUNT(*) n, ROUND(SUM(duration_ns)/1e6,3) ms FROM I WHERE start_ts < {t0} + 500000
GROUP BY 1,2 ORDER BY ms DESC LIMIT 8""")
