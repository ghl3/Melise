# Run docs

One file per model generation: the **recipe** (exact configs, data,
knobs, and why), filled in **before** launch, and the **results**
(wall-clock, metrics, incidents), filled in during/after. The
chronological journal in `../NOTEBOOK.md` references these instead of
restating configs — notebook entries carry the narrative and learnings,
run docs carry the reproducible facts.

| generation | doc | headline |
|---|---|---|
| gen-1 | [gen1-scarlet-harbor.md](gen1-scarlet-harbor.md) | byte-level 17M; test bpb 1.560; first full SFT→GRPO chain |
| gen-2 | [gen2-golden-dell.md](gen2-golden-dell.md) | bpe4k 19M; test bpb 1.247 (−20%); arith cold-start failed generatively |
| gen-3 | [gen3-medium.md](gen3-medium.md) | bpe8k 74M medium on the chat-mix — in prep |
