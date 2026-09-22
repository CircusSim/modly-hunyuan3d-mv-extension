# **modly-hunyuan3d-mv-extension (community fork)**  
Modly extension for **Hunyuan3D 2 Multiview** (tencent/Hunyuan3D-2mv) — Tencent's image-to-3D pipeline variant that conditions on up to 4 reference images (front / back / left / right) instead of a single photo.  
Forked from the structure of [lightningpixel/modly-hunyuan3d-mini-extension](https://github.com/lightningpixel/modly-hunyuan3d-mini-extension "https://github.com/lightningpixel/modly-hunyuan3d-mini-extension"), swapping the single-image mini weights for the multiview weights and adding multi-image input handling.  
## **Status**  
Installs successfully as of this writing. manifest.json was rewritten to match Modly's real schema (confirmed against a working installed extension, hunyuan3d-mini-turbo, rather than guessed). setup.py includes two fixes found during install debugging on real machines — see **Known issues fixed** below.  
Still unverified: the exact shape of multi-image input Modly's UI sends to generator.py ("input": "images" in the manifest is a best guess, since no multi-image example extension existed to check against). If image uploads don't map correctly to the front/back/left/right slots in generator.py's _collect_views(), that's the next thing to check.  
## **What this extension does**  
- Installs an isolated Python venv and installs the right PyTorch build based on the gpu_sm / cuda_version / os / arch info Modly passes in  
- Loads the Hunyuan3D-2mv shape pipeline, and the texture pipeline when the optional native texture extensions are available  
- Accepts 1–4 reference images (front required, back/left/right optional) and feeds all provided views into the multiview conditioning input, for both shape and texture generation  
## **Known issues fixed**  
Modly bundles a minimal/embedded Python build, which caused two install failures that a normal system Python wouldn't hit:  
1. **ensurepip** ** missing** — venv.EnvBuilder(with_pip=True) failed with a nested "command not found" (exit 127). Fixed by creating the venv without pip and bootstrapping it manually via get-pip.py.  
2. **Missing ** **libpython3.11.so.1.0** ** in the venv** — the venv's python binary uses a relative RPATH ($ORIGIN/../lib) to find its shared library at runtime, but venv.EnvBuilder doesn't copy that file into the new venv. Fixed by copying/symlinking it from the base interpreter's lib directory into <venv>/lib after venv creation.  
Also switched from pip install git+https://github.com/Tencent/Hunyuan3D-2.git to downloading a source zip directly, since one test machine (Windows) didn't have git installed/on PATH and this removes that dependency entirely.  
If you hit a fresh install error, delete the partially-created venv/ folder inside the extension's install directory before retrying — this repo's setup.py treats the venv as already-complete simply because the folder exists, so a half-finished one left over from a failed run will not get rebuilt on its own.  
## **Installing**  
3. In Modly, go to **Models → Install from GitHub**.  
4. Point it at this repo's URL: https://github.com/CircusSim/modly-hunyuan3d-mv-extension  
5. Download the model when prompted — this triggers setup.py, which creates the venv and downloads tencent/Hunyuan3D-2mv weights (multi-GB download, budget time and disk space).  
## **Troubleshooting**  
- If install fails partway through, delete the extension's venv/ directory and remove/re-add the extension in Modly rather than just retrying — a partial venv can leave things in a broken state that a simple retry won't fix.  
- If texture generation silently falls back to shape-only, it means the optional native texture-generation extensions aren't available in your environment — this mirrors the mini extension's behavior.  
-    
