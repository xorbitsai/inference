# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Live integration test for Xavier V1 PD (prefill/decode) separation.

This is NOT a unit test. It drives a running Xinference PD deployment over
HTTP and (optionally) inspects the backend debug log to prove that a chat
request is genuinely served by the PD path: the prefill replica stages KV
blocks through the Xavier V1 connector, and the decode replica consumes them
and produces the final answer.

Run it against a live endpoint, e.g. inside/next to the serving host:

    XINF_ENDPOINT=http://10.94.255.37:9999 \
    XINF_API_KEY=sk-xxxx \
    XINF_MODEL_UID=qwen3-pd \
    XINF_LOG_CMD='docker exec xinf tail -n +{start} /root/.xinference/logs/<session>/xinference.log' \
    python test_pd_separation_live.py

`XINF_LOG_CMD` is optional. When provided, the test correlates the backend log
delta for each request and asserts PD-separation markers. `{start}` is
substituted with the log line offset captured just before the request; the
command must print all log lines from that offset onward. Without it, the test
still checks the functional behaviour (a coherent, multi-token answer).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

ENDPOINT = os.environ.get("XINF_ENDPOINT", "http://127.0.0.1:9999").rstrip("/")
API_KEY = os.environ.get("XINF_API_KEY", "")
MODEL_UID = os.environ.get("XINF_MODEL_UID", "qwen3-pd")
LOG_CMD = os.environ.get("XINF_LOG_CMD", "")
LOG_LINECOUNT_CMD = os.environ.get("XINF_LOG_LINECOUNT_CMD", "")
REQUEST_TIMEOUT = float(os.environ.get("XINF_REQUEST_TIMEOUT", "60"))


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _log_baseline():
    if not LOG_LINECOUNT_CMD:
        return None
    try:
        out = subprocess.check_output(
            LOG_LINECOUNT_CMD, shell=True, text=True, timeout=30
        )
        return int(out.strip().split()[0])
    except Exception as e:  # noqa
        print(f"  [warn] failed to read log baseline: {e}")
        return None


def _log_delta(start):
    if not LOG_CMD or start is None:
        return ""
    cmd = LOG_CMD.replace("{start}", str(start + 1))
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=30)
    except Exception as e:  # noqa
        print(f"  [warn] failed to read log delta: {e}")
        return ""


def _chat(stream, prompt, max_tokens=32):
    body = json.dumps(
        {
            "model": MODEL_UID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": stream,
        }
    ).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions", data=body, headers=_headers()
    )
    t0 = time.time()
    text_parts = []
    chunk_count = 0
    error = None
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            if stream:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    chunk_count += 1
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {})
                        text_parts.append(delta.get("content") or "")
                    except Exception:  # noqa
                        pass
            else:
                data = json.loads(resp.read().decode("utf-8", "replace"))
                text_parts.append(data["choices"][0]["message"]["content"])
                chunk_count = 1
    except Exception as e:  # noqa
        error = str(e)
    return {
        "text": "".join(text_parts),
        "chunks": chunk_count,
        "elapsed": round(time.time() - t0, 2),
        "error": error,
    }


def _assert_pd_markers(log_text):
    """Return (ok, details) for PD-separation evidence in the backend log."""
    # PD separation is proven by: the producer staged KV through the Xavier
    # connector, and the decode replica ran generation. ("Chat finished" is
    # NOT used: the prefill sub-call's finish log does not reliably fall inside
    # a single request's captured window, especially for non-streaming.)
    checks = {
        "prefill staged KV (Stage Xavier V1 blocks)": "Stage Xavier V1 blocks"
        in log_text,
        "decode loaded remote KV (Load Xavier V1 blocks)": "Load Xavier V1 blocks"
        in log_text,
    }
    # Any hard error surfaced in the window fails the request.
    hard_errors = [
        marker
        for marker in ("Traceback", "async for", "Engine core initialization failed")
        if marker in log_text
    ]
    return checks, hard_errors


def run_case(name, stream, prompt):
    print(f"\n=== case: {name} (stream={stream}) ===")
    base = _log_baseline()
    res = _chat(stream, prompt)
    log_text = _log_delta(base)

    ok = True
    if res["error"]:
        print(f"  [FAIL] request error: {res['error']}")
        ok = False
    else:
        print(
            f"  response ({res['chunks']} chunk(s), {res['elapsed']}s): "
            f"{res['text']!r}"
        )
        # PD decode must actually produce content — not just prefill's 1 token.
        if not res["text"].strip():
            print("  [FAIL] empty response (decode produced no tokens)")
            ok = False
        if stream and res["chunks"] <= 1:
            print("  [FAIL] streaming produced <=1 chunk (decode likely hung)")
            ok = False

    if log_text:
        checks, hard_errors = _assert_pd_markers(log_text)
        for label, passed in checks.items():
            print(f"  [{'ok' if passed else 'MISS'}] {label}")
            if not passed:
                ok = False
        for err in hard_errors:
            print(f"  [FAIL] backend log contains error marker: {err!r}")
            ok = False
    elif LOG_CMD:
        print("  [warn] no log delta captured; skipped PD-marker assertions")

    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(
        f"endpoint={ENDPOINT} model={MODEL_UID} "
        f"log_checks={'on' if LOG_CMD else 'off'}"
    )
    results = []
    results.append(run_case("streaming chat", True, "用一句话介绍杭州"))
    results.append(run_case("non-streaming chat", False, "用一句话介绍杭州"))

    passed = sum(1 for r in results if r)
    print(f"\n==== {passed}/{len(results)} cases passed ====")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
