#!/usr/bin/env python3
"""Verify repeated greedy outputs and 8/16/32-to-64 prefix consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def generate(base_url: str, input_ids: list[int], max_new_tokens: int) -> list[int]:
    response = post_json(
        base_url.rstrip("/") + "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
    )
    return [int(value) for value in response["output_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=[],
        help="Run only this prompt ID; may be supplied more than once.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    selected_prompt_ids = set(args.prompt_id)
    prompts = [
        prompt
        for prompt in corpus["prompts"]
        if not selected_prompt_ids or prompt["id"] in selected_prompt_ids
    ]
    missing_prompt_ids = selected_prompt_ids - {prompt["id"] for prompt in prompts}
    if missing_prompt_ids:
        raise SystemExit(
            "unknown prompt ID(s): " + ", ".join(sorted(missing_prompt_ids))
        )
    rows = []
    for prompt in prompts:
        started = time.perf_counter()
        runs64 = [generate(args.base_url, prompt["input_ids"], 64) for _ in range(args.repeats)]
        short = {str(length): generate(args.base_url, prompt["input_ids"], length) for length in (8, 16, 32)}
        repeat_exact = all(run == runs64[0] for run in runs64[1:])
        prefix_exact = all(short[str(length)] == runs64[0][:length] for length in (8, 16, 32))
        rows.append(
            {
                "prompt_id": prompt["id"],
                "repeat_exact": repeat_exact,
                "prefix_exact": prefix_exact,
                "hashes64": [hashlib.sha256(json.dumps(run).encode()).hexdigest() for run in runs64],
                "short_outputs": short,
                "reference64": runs64[0],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(prompt["id"], repeat_exact, prefix_exact, flush=True)
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "corpus_sha256": corpus["sha256"],
        "prompt_ids": [prompt["id"] for prompt in prompts],
        "repeats": args.repeats,
        "temperature": 0,
        "protocol_seed": 0,
        "all_repeat_exact": all(row["repeat_exact"] for row in rows),
        "all_prefix_exact": all(row["prefix_exact"] for row in rows),
        "rows": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if not payload["all_repeat_exact"] or not payload["all_prefix_exact"]:
        raise SystemExit("NONDETERMINISM")


if __name__ == "__main__":
    main()
