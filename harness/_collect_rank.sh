#!/bin/zsh
# Measure collection counts for _core test files: real numpy vs port.
cd /tmp/rnp-wt-collect
FILES="$@"
for f in ${=FILES}; do
  real=$(.venv/bin/python -m pytest upstream/numpy/_core/tests/$f -q -p no:cacheprovider --continue-on-collection-errors -c harness/pytest.ini --rootdir upstream/numpy/_core/tests --import-mode=importlib --confcutdir=upstream/numpy/_core/tests --collect-only 2>&1 | tail -3 | tr '\n' ' ')
  port=$(PYTHONPATH=shim:harness/_redirect .venv/bin/python -m pytest upstream/numpy/_core/tests/$f -q -p no:cacheprovider --continue-on-collection-errors -c harness/pytest.ini --rootdir upstream/numpy/_core/tests --import-mode=importlib --confcutdir=upstream/numpy/_core/tests --collect-only 2>&1 | tail -3 | tr '\n' ' ')
  echo "=== $f"
  echo "  REAL: $real"
  echo "  PORT: $port"
done
