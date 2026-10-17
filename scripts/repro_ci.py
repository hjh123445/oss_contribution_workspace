"""Final validation with the CI model pair (lmsys draft), CI-style flags."""
import argparse, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--mem-frac", default="0.7")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import sglang as sgl

    llm = sgl.Engine(
        model_path=args.model,
        dtype=args.dtype,
        attention_backend="triton",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=args.draft,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=2,
        mem_fraction_static=float(args.mem_frac),
        log_level="info",
    )

    info = llm.get_server_info()
    kv = getattr(info, "max_total_num_tokens", None)
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
