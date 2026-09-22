# modly-hunyuan3d-mv-extension (community fork)

Modly extension for **Hunyuan3D 2 Multiview** (`tencent/Hunyuan3D-2mv`) —
Tencent's image-to-3D pipeline variant that conditions on up to 4 reference
images (front / back / left / right) instead of a single photo.

Forked from the structure of
[lightningpixel/modly-hunyuan3d-mini-extension](https://github.com/lightningpixel/modly-hunyuan3d-mini-extension),
swapping the single-image mini weights for the multiview weights and adding
multi-image input handling.

## ⚠️ Status: unverified scaffold

I could not pull the exact source of the original `generator.py` (it wasn't
reachable through search/fetch when this was built), so this fork was
reconstructed from:
- the extension contract Modly's README documents (`manifest.json` +
  `generator.py`, `setup.py` platform-detection pattern)
- Hunyuan3D-2's own documented multiview pipeline usage

Before relying on this, check against your installed Modly version:
- **How Modly passes multi-image input to an extension.** This assumes
  Modly calls `run(images={"front": path, "left": path, ...}, ...)`. If
  Modly's actual multi-image contract looks different (e.g. an ordered
  list instead of a slot dict), update `_collect_views()` in
  `generator.py` accordingly.
- **The `load()`/`run()` signature Modly's runtime expects.** Match
  against the real mini extension's `generator.py` if you can get a copy
  (e.g. by installing the official extension once and reading the cached
  copy from Modly's extensions folder locally, or asking in the Modly
  Discord).
- **`manifest.json`'s `input` block** — this is a guess at how Modly
  would declare "this extension wants N images, not 1." If Modly has an
  existing schema field for that, use it instead.

## What this extension does

- Installs an isolated Python environment (same pattern as the mini
  extension: creates `venv/`, picks a PyTorch build from the `gpu_sm` /
  `cuda_version` / os / arch info Modly passes in)
- Loads the Hunyuan3D-2mv shape pipeline, and the texture pipeline when
  the optional native texture extensions are available
- Accepts 1–4 reference images (`front` required, `back`/`left`/`right`
  optional) and feeds all provided views into the multiview conditioning
  input, for both shape and texture generation

## Installing

1. In Modly, go to **Models → Install from GitHub**.
2. Point it at this repo's URL once you've pushed it to your own GitHub
   (e.g. `https://github.com/<you>/modly-hunyuan3d-mv-extension`).
3. Download the model when prompted — this will trigger `setup.py`, which
   creates the venv and downloads `tencent/Hunyuan3D-2mv` weights
   (multi-GB download, budget time and disk space).

## Troubleshooting

- If install fails partway through, use Modly's **Repair** action to
  recreate the extension venv, same as with the official extensions.
- If texture generation silently falls back to shape-only, it means the
  optional native texture-generation extensions aren't available in your
  environment — this mirrors the mini extension's behavior.
