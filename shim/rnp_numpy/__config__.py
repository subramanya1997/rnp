"""`numpy.__config__` — the build configuration numpy records.

The port is built by cargo, not meson, so this reports the bare minimum the
upstream tests introspect.
"""

CONFIG = {}


def show(mode="stdout"):
    if mode == "dicts":
        return CONFIG
    print("rnp: a Rust port of numpy; no meson build configuration.")


def _check_pyyaml():
    raise NotImplementedError("numpy.__config__._check_pyyaml is not supported")
