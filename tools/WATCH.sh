#!/bin/bash
# ============================================================================
# XEM TIẾN TRÌNH TRÊN SERVER — chạy BẤT KỲ LÚC NÀO, ở terminal/tmux pane khác.
#   bash tools/WATCH.sh          # chụp 1 lần
#   bash tools/WATCH.sh -f       # theo dõi liên tục (Ctrl+C để thoát)
# Không đụng gì tới job đang chạy (chỉ đọc).
# ============================================================================
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
show() {
  clear 2>/dev/null
  echo "════════ $(date '+%F %H:%M:%S') ════════"
  # --- GPU
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    | awk -F', ' '{printf "GPU %s: %-10s/%-10s  util %s\n",$1,$2,$3,$4}'
  # --- job đang chạy
  echo "──────── Job ────────"
  local found=0
  for p in $(pgrep -f "simple_trainer|train.py|render_test_poses|enhance_net|score_local" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    [ -z "$cmd" ] && continue
    found=1
    et=$(ps -o etime= -p $p 2>/dev/null | tr -d ' ')
    name=$(echo "$cmd" | grep -oE "result-dir [^ ]+|experiment_name=[^ ]+|--out [^ ]+" | head -1 | awk '{print $NF}')
    tool=$(echo "$cmd" | grep -oE "simple_trainer|train\.py|render_test_poses|enhance_net|score_local" | head -1)
    printf "  [%s] chạy được %s  %s\n" "${tool:-?}" "${et:-?}" "$(basename "${name:-}")"
  done
  [ "$found" = 0 ] && echo "  (không có job GPU nào — có thể đã xong hoặc chết)"
  # --- tiến độ step: lấy từ log gate mới nhất
  echo "──────── Tiến độ ────────"
  local lg=$(ls -t /tmp/gate_*.txt /tmp/*_train.log 2>/dev/null | head -1)
  if [ -n "$lg" ]; then
    echo "  log: $lg  ($(stat -c %y "$lg" 2>/dev/null | cut -c12-19) sửa lần cuối)"
    # progress bar dùng \r → đổi thành \n rồi lấy dòng cuối có step
    tr '\r' '\n' < "$lg" 2>/dev/null | grep -aoE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+it/s" | tail -1 \
      | sed 's/^/  step /' || echo "  (chưa thấy step — có thể đang compile/load data)"
    # mốc quan trọng đã in ra
    grep -aE "=====|★|VERDICT|❌|✅ ok|Δ =" "$lg" 2>/dev/null | tail -4 | sed 's/^/  /'
  else
    echo "  (chưa có /tmp/gate_*.txt)"
  fi
  echo "──────── Đĩa ────────"
  df -h "$PROJ" | tail -1 | awk '{print "  trống: "$4" / "$2}'
}
if [ "${1:-}" = "-f" ]; then while true; do show; sleep 30; done; else show; fi
