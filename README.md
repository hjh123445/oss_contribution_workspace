# OSS Contribution Workspace — SGLang #44145

Fix for [sgl-project/sglang#44145](https://github.com/sgl-project/sglang/issues/44145):
EAGLE/MTP speculative decoding starves the target KV pool on untied-embedding
models because the draft's duplicate embed_tokens/lm_head are resident when the
scheduler profiles free GPU memory.

- **reports/** — work report (中文) and PR description (English)
- **patches/** — the fix patch (with and without the new unit test)
- **sglang_fix/** — the three modified/added source files
- **scripts/** — reproduction & validation scripts, run logs, result JSONs
- **issues_archive/** — GitHub issue-research JSON dumps from candidate selection

Fix branch: `hjh123445/sglang@fix/eagle-draft-embed-share-before-kv-sizing`

Key result (A800-80GB, Llama-3.1-8B-Instruct + lmsys EAGLE3 draft,
mem-fraction-static=0.28): KV cache **41128 → 48901 tokens (+18.9%)**, E2E
quality unchanged (5/5), speculative unit tests 8/8.
