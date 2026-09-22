"""
Modly extension generator for Hunyuan3D 2 Multiview (Hunyuan3D-2mv).

This is a fork of lightningpixel/modly-hunyuan3d-mini-extension's generator.py,
adapted to:
  1) load the multiview weights (tencent/Hunyuan3D-2mv) instead of the
     single-image mini weights, and
  2) accept up to 4 named reference images (front/back/left/right) instead
     of a single image, wiring them into the pipeline's multiview
     conditioning input.

NOTE: this was reconstructed from the public extension contract Modly
documents (manifest.json + generator.py) and the upstream Hunyuan3D-2
project's documented multiview usage -- not from the original mini
extension's exact source, which wasn't accessible to fetch. Validate
against your installed Modly version before relying on it, in particular:
  - the exact class/module names Modly's runtime expects `run()`/`load()`
    to return
  - how Modly serializes multi-image input to the extension (this assumes
    a dict of {slot_name: image_path})
  - background removal / preprocessing conventions the mini extension used
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# rembg is used by the mini extension for background removal; kept here
# for parity so single-view-only inputs behave the same as before.
try:
    from rembg import remove as rembg_remove
except ImportError:
    rembg_remove = None

EXTENSION_DIR = Path(__file__).parent.resolve()
MODEL_REPO = "tencent/Hunyuan3D-2mv"

# Order matters: this is the canonical view order Hunyuan3D-2mv expects.
VIEW_SLOTS = ["front", "back", "left", "right"]


class Hunyuan3DMultiviewGenerator:
    """
    Wraps the Hunyuan3D-2mv shape (and optional texture) pipeline.
    Modly's runtime is expected to instantiate this once (load()) and then
    call run() per generation request.
    """

    def __init__(self):
        self.shape_pipeline = None
        self.texture_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, model_variant: Optional[str] = None, **kwargs):
        """
        Loads the multiview shape pipeline (and texture pipeline, if the
        optional native texture extensions are present -- same pattern as
        the mini extension).
        """
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_REPO,
            subfolder="hunyuan3d-dit-v2-mv",
            device=self.device,
        )

        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                MODEL_REPO,
                subfolder="hunyuan3d-paint-v2-mv",
                device=self.device,
            )
        except Exception as e:
            # Texture pipeline is optional -- native extensions (e.g. custom
            # rasterizer) may not be available on this platform. Shape-only
            # generation still works.
            print(f"[hunyuan3d-mv] texture pipeline unavailable, shape-only mode: {e}")
            self.texture_pipeline = None

        return self

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    def _load_and_clean(self, image_path: str) -> Image.Image:
        img = Image.open(image_path).convert("RGBA")
        if rembg_remove is not None and img.mode != "RGBA":
            img = rembg_remove(img)
        return img

    def _collect_views(self, images: dict) -> dict:
        """
        images: dict mapping slot name -> file path, e.g.
            {"front": "/tmp/front.png", "left": "/tmp/left.png"}
        Only 'front' is required; missing optional slots are simply omitted
        from the conditioning dict (the mv pipeline supports 1-4 views).
        """
        if "front" not in images or not images["front"]:
            raise ValueError("hunyuan3d-mv requires at least a 'front' reference image")

        views = {}
        for slot in VIEW_SLOTS:
            path = images.get(slot)
            if path:
                views[slot] = self._load_and_clean(path)
        return views

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        images: dict,
        output_path: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        octree_resolution: int = 256,
        generate_texture: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        images: {slot_name: file_path}, slot_name in ("front","back","left","right")
        output_path: where to write the final mesh (.glb)
        Returns the output_path on success.
        """
        if self.shape_pipeline is None:
            raise RuntimeError("Generator not loaded -- call load() first")

        views = self._collect_views(images)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        mesh = self.shape_pipeline(
            image=views,  # multiview conditioning: dict of PIL images by view name
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            generator=generator,
        )[0]

        if generate_texture and self.texture_pipeline is not None:
            # Texture pipeline paints using the same set of reference views
            mesh = self.texture_pipeline(mesh, image=views)
        elif generate_texture and self.texture_pipeline is None:
            print("[hunyuan3d-mv] texture requested but pipeline unavailable; exporting shape-only mesh")

        mesh.export(output_path)
        return output_path


# Modly's extension runtime is expected to do roughly:
#   gen = Hunyuan3DMultiviewGenerator().load()
#   gen.run(images={"front": "...", "left": "..."}, output_path="...")
def get_generator():
    return Hunyuan3DMultiviewGenerator()
