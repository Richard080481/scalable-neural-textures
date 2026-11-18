# Dataset format and UV convention

Each split contains identically named `rgb/<view>.png`, `uv/<view>.npy`, and
`mask/<view>.png` files. RGB is converted to `[3,H,W]` float data in `[0,1]`;
mask is `[1,H,W]`; UV is float32 `[H,W,2]` in `(u,v)` order and `[0,1]`.

`u=0` is the left texture edge and `u=1` the right edge. `v=0` is the top and
`v=1` the bottom. The sampler converts `(u,v)` to the `grid_sample` coordinate
`(2u-1,2v-1)`. Set `data.v_flip: true` only for sources whose V axis points up.
Invalid pixels must have mask value zero. UV values at those pixels are ignored
by the masked loss.

The `tests/test_neural_texture.py` corner test is the executable specification
for this convention.

