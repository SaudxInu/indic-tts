# vllm serve \
# --uvicorn-log-level=info \
# "snorbyte/snorTTS-Indic-v0"
# --served-model-name "snorbyte/snorTTS-Indic-v0" "llm" \
# --max-model-len 2048 \
# --max-num-seqs 5 \
# --quantization fp8 \
# --host 0.0.0.0 \
# --port 8000

import time
import statistics
import asyncio
import aiohttp
from typing import Any


async def stream_completion(prompt: str, max_tokens: int = 1950):
    payload: dict[str, Any] = {
        "prompt": prompt,
        "model": "snorbyte/snorTTS-Indic-v0",
        "stream": True,
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.05,
        "add_special_tokens": False,
        "stop_token_ids": [128258],
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with aiohttp.ClientSession(base_url="http://0.0.0.0:8000") as session:
        async with session.post(
            "/v1/completions", json=payload, headers=headers, timeout=60
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                if raw:
                    print(raw.decode("utf-8").strip())

                    if raw.decode("utf-8").strip() == "data: [DONE]":
                        break


async def main():
    times = []
    for _ in range(30):
        start = time.time()
        await stream_completion(
            "<custom_token_3><|begin_of_text|>hindi159: कल मैंने सिर्फ ₹500 में एक cool headphones ले लिए, बहुत बढ़िया deal था यार!<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
        )
        elapsed = time.time() - start
        times.append(elapsed)

    print("\n--- Summary ---")
    print(f"Average: {statistics.mean(times):.3f} s")
    print(f"Std Dev:  {statistics.stdev(times):.3f} s")
    print(f"Min:      {min(times):.3f} s")
    print(f"Max:      {max(times):.3f} s")


if __name__ == "__main__":
    asyncio.run(main())
