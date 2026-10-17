"""Reproduce SGLang #44145: draft embed/lm_head starve target KV pool."""
import argparse, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--mem-frac", default="0.85")
    parser.add_argument("--out", default=None)
    parser.add_argument("--algo", default="EAGLE3")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import sglang as sgl

    llm = sgl.Engine(
        model_path=args.model,
        speculative_algorithm=args.algo,
        speculative_draft_model_path=args.draft,
        speculative_num_steps=3,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=8,
        mem_fraction_static=float(args.mem_frac),
        log_level="info",
    )

    info = llm.get_server_info()
    kv = getattr(info, "max_total_num_tokens", None)
    if kv is None and isinstance(info, dict):
        kv = info.get("max_total_num_tokens")
    print(f"RESULT_MAX_TOTAL_NUM_TOKENS={kv}")

    out = llm.generate(
        ["The capital of France is"],
        sampling_params={"max_new_tokens": 32, "temperature": 0},
    )
    print("RESULT_SAMPLE:", out[0]["text"][:150].replace("\n", " "))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"max_total_num_tokens": kv, "sample": out[0]["text"][:300]}, f, indent=2)

    llm.shutdown()
    print("DONE")

if __name__ == "__main__":
    main()
