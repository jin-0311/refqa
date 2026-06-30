#!/usr/bin/env bash
# Download PMC OA Bulk JATS XML packages (baseline + incremental).
#
# Usage:
#   fetch_pmc_oa.sh [oa_comm|oa_noncomm|oa_other|all]
#
# Idempotent: skips files already present with matching Content-Length.
# Resumable: uses wget --continue. Verifies size after each file.
#
# Output layout:
#   $ROOT/raw/pmc/<subset>/xml/oa_<subset>_xml.*.tar.gz
#   $ROOT/raw/pmc/<subset>/xml/oa_<subset>_xml.*.filelist.csv
#   $ROOT/raw/pmc/manifests/<subset>_index.html       (snapshot of remote listing)
#   $ROOT/raw/pmc/manifests/<subset>_remote.tsv       (filename<TAB>size)
#
# Logs:
#   $ROOT/logs/download/<subset>.log

set -u  # don't set -e; we want the loop to continue on individual failures

ROOT="./data"
LOG_DIR="./logs/download"
BASE_URL="https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk"
MANIFEST_DIR="$ROOT/raw/pmc/manifests"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

fetch_subset() {
  local subset="$1"   # oa_comm / oa_noncomm / oa_other
  local url="$BASE_URL/$subset/xml/"
  local out_dir="$ROOT/raw/pmc/$subset/xml"
  local log_file="$LOG_DIR/${subset}.log"
  local idx="$MANIFEST_DIR/${subset}_index.html"
  local manifest="$MANIFEST_DIR/${subset}_remote.tsv"

  mkdir -p "$out_dir" "$LOG_DIR" "$MANIFEST_DIR"

  # Re-route logging to per-subset file (and also tee stderr for live view)
  exec >>"$log_file" 2>&1

  log "=== fetch_subset start: $subset ==="
  log "remote=$url  out=$out_dir"

  # 1) snapshot remote directory listing
  if ! curl -sS --retry 5 --retry-delay 5 --max-time 60 "$url" -o "$idx"; then
    log "ERROR: failed to fetch index $url"
    return 1
  fi

  # 2) extract candidate filenames (.tar.gz + .filelist.csv + .filelist.txt)
  local files
  files=$(grep -oE 'href="[^"]+\.(tar\.gz|filelist\.csv|filelist\.txt)"' "$idx" \
          | sed 's/href="//;s/"$//' | sort -u)

  local n
  n=$(printf '%s\n' "$files" | grep -c .)
  log "candidates: $n files"

  # 3) build remote manifest with sizes (via HEAD)
  : > "$manifest"
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    local sz
    sz=$(curl -sSI --retry 3 --retry-delay 3 --max-time 30 "$url$f" \
         | awk '/^[Cc]ontent-[Ll]ength: [0-9]+/{print $2}' | tr -d '\r')
    [ -z "$sz" ] && sz=0
    printf '%s\t%s\n' "$f" "$sz" >>"$manifest"
  done <<<"$files"
  log "wrote remote manifest: $manifest ($(wc -l <"$manifest") rows)"

  # 4) download missing or size-mismatched files
  local got=0 skipped=0 failed=0
  while IFS=$'\t' read -r fname remote_size; do
    [ -z "$fname" ] && continue
    local local_path="$out_dir/$fname"
    local local_size=0
    [ -f "$local_path" ] && local_size=$(stat -c %s "$local_path" 2>/dev/null || echo 0)

    if [ "$remote_size" -gt 0 ] && [ "$local_size" -eq "$remote_size" ]; then
      skipped=$((skipped + 1))
      continue
    fi

    log "GET  $fname  (remote=$remote_size local=$local_size)"
    # --continue resumes partial downloads; -q for quiet (logs come from our script)
    if wget --quiet --continue --tries=5 --waitretry=10 --timeout=120 \
        -O "$local_path" "$url$fname"; then
      local new_size
      new_size=$(stat -c %s "$local_path" 2>/dev/null || echo 0)
      if [ "$remote_size" -gt 0 ] && [ "$new_size" -ne "$remote_size" ]; then
        log "WARN size mismatch after download: $fname got=$new_size want=$remote_size"
        failed=$((failed + 1))
      else
        got=$((got + 1))
      fi
    else
      log "ERROR wget failed: $fname"
      failed=$((failed + 1))
    fi
  done <"$manifest"

  log "DONE $subset  downloaded=$got skipped=$skipped failed=$failed"
  return 0
}

main() {
  local target="${1:-all}"
  case "$target" in
    oa_comm|oa_noncomm|oa_other) fetch_subset "$target" ;;
    all)
      # serialize to be FTP-polite; user can also launch in parallel by calling
      # this script three times with different args.
      fetch_subset oa_comm
      fetch_subset oa_noncomm
      fetch_subset oa_other
      ;;
    *)
      echo "usage: $0 [oa_comm|oa_noncomm|oa_other|all]" >&2
      exit 2
      ;;
  esac
}

main "$@"
