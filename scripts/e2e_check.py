"""E2E quality check after eager embed sharing: multi-request, longer gen."""
import os
def main():
    W = os.environ["W"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    import sglang as sgl
    llm = sgl.Engine(
        model_path=f"{W}/sglang/ms_cache/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master",
        dtype="float16",
        attention_backend="triton",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=f"{W}/models/eagle3_draft_lmsys",
        speculative_num_steps=3,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=8,
        mem_fraction_static=0.85,
        log_level="warn",
    )
    prompts = [
        "The capital of France is",
        "List three primary colors:",
        "Water boils at",
        "The largest planet in the solar system is",
        "Complete: 2 + 2 =",
    ]
    outs = llm.generate(prompts, sampling_params={"max_new_tokens": 48, "temperature": 0})
    ok = 0
    checks = ["Paris", "red", "100", "Jupiter", "4"]
    for o, c in zip(outs, checks):
        t = o["text"]
        hit = c.lower() in t.lower()
        ok += hit
        print(f"[{'OK ' if hit else 'FAIL'}] expect~{c!r}: {t[:90]!r}")
    print(f"SCORE={ok}/{len(checks)}")
    llm.shutdown()

if __name__ == "__main__":
    main()
