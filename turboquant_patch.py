"""
Patches vLLM's TurboQuant attention backend to fix the workspace allocation
failure when MTP speculative decoding is active.

Root cause: during CUDA graph warmup the MTP drafter's dummy_run passes
attn_metadata=None, which returns early (line 410 of turboquant_attn.py)
before the workspace allocation code runs. The workspace is therefore never
sized for the drafter's decode path. On the first real decode the workspace
is already locked at 0 bytes and raises AssertionError.

Fix: catch AssertionError and fall back to persistent per-layer buffers
allocated once per TurboQuantAttentionImpl instance and reused every step.
"""
import sys
import vllm.v1.attention.backends.turboquant_attn as _mod
import inspect

_path = inspect.getfile(_mod)
with open(_path) as f:
    _src = f.read()

if "_fallback_mid_o" in _src:
    print(f"turboquant_patch: already applied to {_path}")
    sys.exit(0)

_OLD = (
    "        if is_workspace_manager_initialized():\n"
    "            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.\n"
    "            mid_o_buf, output_buf, lse_buf = (\n"
    "                current_workspace_manager().get_simultaneous(\n"
    "                    ((B, Hq, S, D + 1), torch.float32),\n"
    "                    ((B, Hq, D), query.dtype),\n"
    "                    ((B, Hq), torch.float32),\n"
    "                )\n"
    "            )"
)

_NEW = (
    "        if is_workspace_manager_initialized():\n"
    "            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.\n"
    "            # Falls back to per-layer persistent buffers if workspace is locked and\n"
    "            # insufficiently sized (MTP drafter decode not profiled during warmup).\n"
    "            try:\n"
    "                mid_o_buf, output_buf, lse_buf = (\n"
    "                    current_workspace_manager().get_simultaneous(\n"
    "                        ((B, Hq, S, D + 1), torch.float32),\n"
    "                        ((B, Hq, D), query.dtype),\n"
    "                        ((B, Hq), torch.float32),\n"
    "                    )\n"
    "                )\n"
    "            except AssertionError:\n"
    "                if (\n"
    "                    not hasattr(self, '_fallback_mid_o')\n"
    "                    or self._fallback_mid_o.shape[0] < B\n"
    "                ):\n"
    "                    device = query.device\n"
    "                    self._fallback_mid_o = torch.empty(\n"
    "                        (B, Hq, S, D + 1), dtype=torch.float32, device=device\n"
    "                    )\n"
    "                    self._fallback_output = torch.empty(\n"
    "                        (B, Hq, D), dtype=query.dtype, device=device\n"
    "                    )\n"
    "                    self._fallback_lse = torch.empty(\n"
    "                        (B, Hq), dtype=torch.float32, device=device\n"
    "                    )\n"
    "                mid_o_buf = self._fallback_mid_o[:B]\n"
    "                output_buf = self._fallback_output[:B]\n"
    "                lse_buf = self._fallback_lse[:B]"
)

if _OLD not in _src:
    print(f"turboquant_patch: WARNING — expected pattern not found in {_path}")
    print("vLLM version may differ from 0.21.0. Skipping patch.")
    sys.exit(0)

_patched = _src.replace(_OLD, _NEW, 1)
with open(_path, "w") as f:
    f.write(_patched)
print(f"turboquant_patch: applied to {_path}")
