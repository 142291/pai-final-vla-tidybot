#!/usr/bin/env bash
# モーターバス基板が USB シリアルとして見えているかだけ確認する(基板 2 枚なら 2 行出る)。
set -euo pipefail
echo "== /dev/tty.usbmodem* / usbserial* =="
ls /dev/tty.usbmodem* /dev/tty.usbserial* 2>/dev/null || echo "(見つからない: USB ケーブル・基板の電源・ジャンパ B 側を確認)"
echo
echo "== USB デバイス一覧(シリアル系) =="
system_profiler SPUSBDataType 2>/dev/null | grep -iE "serial|ch34|cp210|ftdi|waveshare|usb modem|Product ID|^\s{8}[A-Za-z].*:$" | head -30
