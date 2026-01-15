#!/bin/bash

set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR=""


usage() {
    echo "Usage: $0 -i <input_pcap_dir> -o <output_csv_dir>"
    echo "  -i : Input directory containing .pcap or .pcapng files (required)"
    echo "  -o : Output directory to store CSV files (required)"
    echo "Example: $0 -i /mnt/data/pcaps -o /mnt/data/csvs"
    exit 1
}


while getopts "i:o:h" opt; do
    case $opt in
        i) INPUT_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "[ERROR] Both -i and -o are required."
    usage
fi


if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERROR] Input directory does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

log() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*"
}


find "$INPUT_DIR" -type f \( -iname "*.pcap" -o -iname "*.pcapng" \) | while read -r pcap_file; do

    rel_path="${pcap_file#$INPUT_DIR/}"
   
    csv_rel_path="${rel_path%.*}.csv"
   
    csv_file="$OUTPUT_DIR/$csv_rel_path"

   
    mkdir -p "$(dirname "$csv_file")"

   
    if [[ -f "$csv_file" ]]; then
        log "Skipped (already exists): $csv_file"
        continue
    fi

    log "Processing: $pcap_file -> $csv_file"

   
    tshark -r "$pcap_file" -Y "tcp or udp" -T fields \
        -e frame.len \
        -e ip.src -e ip.dst \
        -e tcp.srcport -e tcp.dstport \
        -e udp.srcport -e udp.dstport \
        -E separator=, -E quote=d \
        > "$csv_file" 2>/dev/null

    if [[ ! -s "$csv_file" ]]; then
        log "Warning: No TCP/UDP traffic or empty output for $pcap_file"
    
    fi
done

