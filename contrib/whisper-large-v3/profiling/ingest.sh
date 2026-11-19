#!/bin/bash
# ingest.sh <neff> <ntff> <name> -- neuron-explorer parquet ingest (no server) into $ASR_WORK/profiles/ne-data.
# Large decoder-step traces take several minutes; bounds.py / attrib.py / deep.py read the parquet afterwards.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
mkdir -p "$ASR_WORK/profiles"
neuron-explorer view -n "$1" -s "$2" --data-path "$ASR_WORK/profiles/ne-data" --display-name "$3" --ingest-only \
  > "$ASR_WORK/profiles/ingest_$3.log" 2>&1
echo "$3 rc=$?" >> "$ASR_WORK/profiles/ingest_done.log"
