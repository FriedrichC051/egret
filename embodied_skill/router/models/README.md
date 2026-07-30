# Model weights

This directory holds the two small trainable components of the router. They are
**not** shipped with the repository and must be produced by the training scripts
(a few minutes each on a single 8 GB GPU):

| File          | Produced by                              | Used by                          |
|---------------|------------------------------------------|----------------------------------|
| `l0_head.pt`  | `python embodied_skill/router/code/l0_head.py`          | Goal Parser head (methods `m1h`, `m1h4`) |
| `l4_head.pt`  | `python embodied_skill/router/code/l4_head.py --l0 head`| Effect-grounded tie-break head (`m1h4`)  |

The frozen encoder (`bge-small-en-v1.5`) is downloaded automatically from the
Hugging Face Hub on first use, or can be placed here as `bge-small-en-v1.5/`.

See the top-level `README.md` (section "Full reproduction") for details.
