#!/bin/zsh
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  print -u2 "Refusing to run: this test suite requires macOS."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  print -u2 "Refusing to run: this test suite requires Apple Silicon (arm64)."
  exit 1
fi

metal_report="$(system_profiler SPDisplaysDataType)"
if ! grep -Eq "Metal Support: Metal" <<< "$metal_report"; then
  print -u2 "Refusing to run: Metal support was not detected."
  exit 1
fi

chip="$(awk -F': ' '/Chipset Model:/ {print $2; exit}' <<< "$metal_report")"
metal="$(awk -F': ' '/Metal Support:/ {print $2; exit}' <<< "$metal_report")"
print "Verified test host: macOS $(sw_vers -productVersion), ${chip}, ${metal}."
