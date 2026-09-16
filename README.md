# FlyVoxelisedReality

**Reality Voxelised Through a Fly** — an experimental real-time graphics prototype exploring a pipeline from live computer vision to an OpenGL voxel world, fly-inspired perception, mapped Drosophila neural activity, and learned/neural appearance.

> This repository is an independent technical prototype used to validate the concept before any course group project is scoped. It is intentionally narrower than a complete game.

## What works now

### M0 — live CV → OpenGL voxels ✅

```text
webcam
  ↓
OpenCV preprocessing + optical flow
  ↓
1,728 live voxel instances
  ↓
OpenGL instanced rendering
```

M0 has been validated on Windows with an RTX 2080 and a live webcam.

### M1 — monocular relative depth → spatial voxel reconstruction ✅

M1 replaces the original brightness-as-depth experiment with a real monocular depth network:

```text
webcam RGB
   ↓
MiDaS v2.1 Small (OpenCV DNN)
   ↓
relative inverse-depth map
   ↓
pinhole back-projection
   ↓
quantised 3D voxel positions
   ↓
OpenGL instanced rendering
```

M1 has been locally validated with a live webcam. Foreground human geometry and room background separate spatially in the reconstructed voxel field.

Important: MiDaS provides **relative monocular depth**, not calibrated metric distance in metres. The resulting scene has real depth ordering and perspective structure, but it is not yet an accurate metric scan of the room.

Optical flow runs alongside depth and is visualised as cyan/green activity on moving voxels.

### M2 — fly-inspired perception renderer 🚧

M2 adds a second OpenGL render stage:

```text
voxel scene
   ↓
off-screen framebuffer (colour + depth)
   ↓
full-screen GLSL perception pass
   ↓
staggered ommatidial sampling + wide FOV + visible-spectrum remap
```

Press `F` to toggle between:

- **RESEARCHER** — conventional OpenGL view
- **FLY** — fly-inspired compound-eye preview

The fly view is deliberately labelled **fly-inspired**, not biologically exact. A normal webcam does not capture ultraviolet light and the current renderer does not model individual Drosophila photoreceptor classes or retinal neural processing. The colour transform is therefore only a visible-RGB proxy, while the staggered lens field is a graphics representation of compound-eye angular sampling.

This renderer is intended to become the visual front end for M3, where real-time motion/features will drive a mapped Drosophila visual pathway.

## Planned full pipeline

```text
LIVE CAMERA
    ↓
REAL-TIME CV
(depth / segmentation / optical flow)
    ↓
VOXELISED REALITY
(OpenGL)
    ↓
FLY PERCEPTION
(compound-eye / motion channels)
    ↓
FLY NEURAL MAP
(mapped visual-pathway activation)
    ↓
NEURAL APPEARANCE
(learned material / reconstruction pass)
```

## Controls

- `Esc` — quit
- `Space` — pause/resume camera-driven voxel updates
- `D` — toggle monocular depth vs the old brightness fallback
- `F` — toggle researcher vs fly-inspired perception
- `R` — reset camera orbit
- Drag with **left mouse** — orbit around the reconstructed scene
- Mouse wheel — zoom

## Dependencies

- C++20 compiler
- OpenGL 3.3+
- CMake 3.24+ (Visual Studio 2026 users need a recent CMake that knows the VS 18 generator)
- webcam
- [vcpkg](https://github.com/microsoft/vcpkg)

The manifest installs GLEW, GLFW, GLM and OpenCV including the DNN module.

## Get the M1 depth model

The model weights are deliberately **not committed** to this repository.

From PowerShell in the repository root:

```powershell
.\scripts\download_depth_model.ps1
```

This downloads the official MiDaS v2.1 Small ONNX model to:

```text
models/model-small.onnx
```

Source: `isl-org/MiDaS`, MiDaS v2.1 release.

If the model is absent or depth inference fails, the application remains runnable and falls back to the original image-intensity depth experiment. The window title will say `FALLBACK` rather than `DEPTH`.

## Build

Install dependencies through the manifest:

```powershell
vcpkg install
```

Configure and build with your vcpkg toolchain. Example for Visual Studio 2026:

```powershell
cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

cmake --build build --config Release
```

Run from the repository root so the default `models/model-small.onnx` path resolves:

```powershell
.\build\Release\FlyVoxelisedReality.exe
```

You can also supply a model path explicitly:

```powershell
.\build\Release\FlyVoxelisedReality.exe "D:\models\model-small.onnx"
```

## Milestones

- [x] Repository + architecture scaffold
- [x] **M0:** webcam → OpenCV → live OpenGL voxel field
- [x] **M1:** relative monocular depth → spatial voxel reconstruction
- [ ] **M2:** fly-inspired compound-eye / motion perception (implemented; awaiting local validation)
- [ ] **M3:** real Drosophila visual-pathway map + live activation
- [ ] **M4:** researcher ↔ fly dual perspective
- [ ] **M5:** bounded neural-rendering experiment
- [ ] **M6:** public demo video + technical write-up

## Design rule

The neural component must **augment rather than replace the OpenGL renderer**. Geometry, camera, buffers, transforms, instancing, shading, render targets, and interaction remain explicit graphics work.
