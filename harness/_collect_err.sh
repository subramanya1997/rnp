#!/bin/zsh
cd /tmp/rnp-wt-collect
for f in "$@"; do
  echo "########## $f"
  PYTHONPATH=shim:harness/_redirect .venv/bin/python -m pytest upstream/numpy/_core/tests/$f -q -p no:cacheprovider --continue-on-collection-errors -c harness/pytest.ini --rootdir upstream/numpy/_core/tests --import-mode=importlib --confcutdir=upstream/numpy/_core/tests --collect-only 2>&1 | grep -v '^$' | tail -25
done
