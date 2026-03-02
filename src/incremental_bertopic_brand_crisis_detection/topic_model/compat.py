from __future__ import annotations

import sys
import types


def ensure_llama_cpp_importable() -> None:
    """
    Ensure `import llama_cpp` does not crash BERTopic import.

    Some environments have a partially working llama_cpp install that raises
    runtime loader errors at import time (for example missing libcublas.so.13).
    BERTopic imports its LlamaCPP representation module eagerly, so this can
    block all BERTopic usage even when LlamaCPP is not used.

    We install a lightweight stub only when import fails, which keeps
    non-LlamaCPP BERTopic workflows functional.
    """

    try:
        import llama_cpp  # noqa: F401

        return
    except Exception:
        pass

    module = types.ModuleType("llama_cpp")

    class Llama:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "llama_cpp is unavailable in this environment. "
                "Install a compatible llama_cpp build to use LlamaCPP representation."
            )

    module.Llama = Llama
    sys.modules["llama_cpp"] = module
