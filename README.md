# FlyVoxelisedReality

**Reality Voxelised Through a Fly** — an experimental real-time graphics prototype exploring a pipeline from live computer vision to an OpenGL voxel world, a fly agent, mapped Drosophila neural activity, and later learned/neural rendering.

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

Validated on Windows with an RTX 2080 and a live webcam.

### M1 — monocular relative depth → spatial voxel reconstruction ✅

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

MiDaS provides **relative monocular depth**, not calibrated metric distance in metres. The result preserves useful depth ordering and perspective structure, but it is not a metric room scan.

### M2 — fly agent + realtime visual stimulus 🚧

The discarded screen-space “compound eye” filter has been removed. M2 now keeps the researcher view and places a small fly agent directly inside the reconstructed voxel world.

```text
webcam
  ↓
optical flow + depth
  ↓
motion centroid in 3D
  ↓
retina → lamina → medulla → lobula
  ↓
prototype steering response
  ↓
fly moves inside the OpenGL voxel scene
```

The four named stages are real major parts of the fly visual system, but the current scalar transfer functions and steering policy are deliberately **engineering placeholders**, not a claim of biologically faithful neural simulation. M3 will replace this proxy with connectivity grounded in a real Drosophila visual-neural map.

Four small activity nodes float above the fly in the order `retina → lamina → medulla → lobula`; their activation follows the live motion stimulus. The fly body uses the same OpenGL instanced-cube renderer as the reconstructed world.

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
FLY AGENT
    ↓
FLY NEURAL MAP
(mapped visual-pathway activation)
    ↓
BEHAVIOUR
    ↓
NEURAL APPEARANCE
(learned rendering / reconstruction experiment)
```

## Controls

- `Esc` — quit
- `Space` — pause/resume camera-driven updates
- `A` — enable/disable fly steering while keeping neural activity visible
- `D` — toggle monocular depth vs the old brightness fallback
- `R` — reset researcher camera orbit
- Drag with **left mouse** — orbit around the reconstructed scene
- Mouse wheel — zoom

## Dependencies

- C++20 compiler
- OpenGL 3.3+
- CMake 3.24+ (Visual Studio 2026 users need a recent CMake that knows the VS 18 generator)
- webcam
- [vcpkg](https://github.com/microsoft/vcpkg)

The manifest installs GLEW, GLFW, GLM and OpenCV including the DNN module.

## Get the depth model

The model weights are deliberately **not committed** to this repository.

From PowerShell in the repository root:

```powershell
.\scripts\download_depth_model.ps1
```

This downloads the official MiDaS v2.1 Small ONNX model to:

```text
models/model-small.onnx
```

If the model is absent or depth inference fails, the application stays runnable using the original image-intensity fallback.

## Build

```powershell
vcpkg install

cmake -S . -B build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

cmake --build build --config Release
.\build\Release\FlyVoxelisedReality.exe
```

## Milestones

- [x] Repository + architecture scaffold
- [x] **M0:** webcam → OpenCV → live OpenGL voxel field
- [x] **M1:** relative monocular depth → spatial voxel reconstruction
- [ ] **M2:** fly agent + realtime visual stimulus (implemented; awaiting local validation)
- [ ] **M3:** real Drosophila visual-pathway map + live activation
- [ ] **M4:** researcher ↔ fly interaction / inspection modes
- [ ] **M5:** bounded neural-rendering experiment
- [ ] **M6:** public demo video + technical write-up

## Design rule

The neural component must **augment rather than replace the OpenGL renderer**. Geometry, camera, buffers, transforms, instancing, shading, render targets, and interaction remain explicit graphics work.
