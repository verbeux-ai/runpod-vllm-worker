#!/usr/bin/env python3
"""
SGLang launcher — espelha 100% o /opt/vllm/run_sglang.sh do GCP sglang-lab-001.

Modos:
  --vast-bg     SGLang em background na porta MODEL_PORT (entrypoint roda PyWorker em paralelo)
  --standalone  SGLang em foreground

Todas as env vars opcionais. MODEL_ID obrigatória.
"""
import os
import sys

MODEL_ID    = os.environ["MODEL_ID"]
SERVED_NAME = os.environ.get("SERVED_NAME", MODEL_ID.split("/")[-1].lower())
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

# Configs default = espelham GCP run_sglang.sh (lab-validated)
TP_SIZE                  = os.environ.get("TENSOR_PARALLEL_SIZE",      "1")
CONTEXT_LENGTH           = os.environ.get("MAX_MODEL_LEN",             "262144")
MEM_FRACTION             = os.environ.get("MEM_FRACTION_STATIC",       "0.85")
CHUNKED_PREFILL          = os.environ.get("CHUNKED_PREFILL_SIZE",      "2096")
MAX_RUNNING              = os.environ.get("MAX_RUNNING_REQUESTS",      "12")
QUANTIZATION             = os.environ.get("QUANTIZATION",              "modelopt")
KV_CACHE_DTYPE           = os.environ.get("KV_CACHE_DTYPE",            "fp8_e4m3")
REASONING_PARSER         = os.environ.get("REASONING_PARSER",          "qwen3")
TOOL_CALL_PARSER         = os.environ.get("TOOL_CALL_PARSER",          "qwen3_coder")
SPEC_ALGORITHM           = os.environ.get("SPECULATIVE_ALGORITHM",     "NEXTN")
SPEC_NUM_STEPS           = os.environ.get("SPECULATIVE_NUM_STEPS",     "3")
SPEC_EAGLE_TOPK          = os.environ.get("SPECULATIVE_EAGLE_TOPK",    "1")
SPEC_NUM_DRAFT_TOKENS    = os.environ.get("SPECULATIVE_NUM_DRAFT_TOKENS", "4")
MAMBA_SCHEDULER          = os.environ.get("MAMBA_SCHEDULER_STRATEGY",  "extra_buffer")
ATTENTION_BACKEND        = os.environ.get("ATTENTION_BACKEND",         "flashinfer")
API_KEY                  = os.environ.get("SGLANG_API_KEY",            "")
EXTRA_ARGS               = os.environ.get("EXTRA_SGLANG_ARGS",         "")

MODEL_PORT = int(os.environ.get("MODEL_PORT", "30000"))


def resolve_hf_home():
    if os.path.isdir("/models") and os.path.exists("/models/PREBAKED"):
        return "/models"
    if os.path.isdir("/workspace"):
        d = "/workspace/hf-cache"
        os.makedirs(d, exist_ok=True)
        return d
    return os.environ.get("HF_HOME", "/root/.cache/huggingface")


def build_cmd():
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path",                MODEL_ID,
        "--tp-size",                   TP_SIZE,
        "--host",                      "0.0.0.0",
        "--port",                      str(MODEL_PORT),
        "--context-length",            CONTEXT_LENGTH,
        "--mem-fraction-static",       MEM_FRACTION,
        "--chunked-prefill-size",      CHUNKED_PREFILL,
        "--max-running-requests",      MAX_RUNNING,
        "--quantization",              QUANTIZATION,
        "--kv-cache-dtype",            KV_CACHE_DTYPE,
        "--reasoning-parser",          REASONING_PARSER,
        "--tool-call-parser",          TOOL_CALL_PARSER,
        "--speculative-algorithm",     SPEC_ALGORITHM,
        "--speculative-num-steps",     SPEC_NUM_STEPS,
        "--speculative-eagle-topk",    SPEC_EAGLE_TOPK,
        "--speculative-num-draft-tokens", SPEC_NUM_DRAFT_TOKENS,
        "--mamba-scheduler-strategy",  MAMBA_SCHEDULER,
        "--attention-backend",         ATTENTION_BACKEND,
        "--served-model-name",         SERVED_NAME,
        "--enable-metrics",
        "--enable-cache-report",
        "--trust-remote-code",
    ]
    if API_KEY:    cmd += ["--api-key", API_KEY]
    if EXTRA_ARGS: cmd += EXTRA_ARGS.split()
    return cmd


def build_env():
    env = {**os.environ}
    env["HF_HOME"] = resolve_hf_home()
    env["SGLANG_ENABLE_SPEC_V2"] = "1"  # obrigatório pra NEXTN (sem ele cai pra EAGLE = OOM)
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    return env


def exec_sglang_foreground():
    cmd, env = build_cmd(), build_env()
    print(f"[handler] HF_HOME={env['HF_HOME']}", flush=True)
    print(f"[handler] SGLANG_ENABLE_SPEC_V2={env['SGLANG_ENABLE_SPEC_V2']}", flush=True)
    print(f"[handler] exec: {' '.join(cmd)}", flush=True)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--standalone" in args or "--vast-bg" in args:
        exec_sglang_foreground()
    else:
        # default = standalone
        exec_sglang_foreground()
