"""pytest plugin: run only one deterministic shard of a file's collected tests.

Used by harness/run.py to split very large upstream test files (notably
test_multiarray.py, ~14k tests) across several subprocesses so that each one
finishes inside the per-file timeout. This is *scheduling only*: the union of
the shards is exactly the file's full collection, no test is skipped,
deselected-for-good, modified, or weakened. Every collected item lands in
exactly one shard, and harness/run.py sums the shards' results.

Enabled with `-p rnp_shard --rnp-shard=<i>/<n>`.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--rnp-shard", action="store", default=None,
        help="run only shard i of n, as 'i/n' (0 <= i < n)")


def pytest_collection_modifyitems(config, items):
    spec = config.getoption("--rnp-shard")
    if not spec:
        return
    index, total = (int(x) for x in spec.split("/"))
    if total <= 1:
        return
    # Sort by node id first so the partition does not depend on collection
    # order across runs, then take a strided slice: shard i gets items
    # i, i+n, i+2n, ... This interleaves classes, which balances the shards
    # far better than contiguous blocks would.
    ordered = sorted(range(len(items)), key=lambda k: items[k].nodeid)
    keep = {ordered[k] for k in range(index, len(ordered), total)}
    items[:] = [it for k, it in enumerate(items) if k in keep]
