# FlyVoxelisedReality

**Reality Voxelised Through a Fly** — an experimental real-time graphics prototype combining OpenGL voxel rendering, realtime computer vision, a fly agent, mapped Drosophila neural activity, and later bounded neural rendering.

> This repository is an independent technical prototype used to validate the concept before any course group project is scoped. It is intentionally narrower than a complete game.

## Current scene sources

The renderer now has three switchable scene sources:

- `1` — **Office / Studio**: desk, monitor, chair, plant, mug, window and shelving. This is the spatial/navigation showcase.
- `2` — **Kitchen / Fruit Table**: dining table, fruit bowl, chairs, cabinets, plant, mug and pendant light. This is the controlled stimulus / fly-behaviour showcase.
- `3` — **Live Camera Reconstruction**: webcam → monocular depth → realtime voxel reconstruction.

The two preset scenes are built procedurally from the same OpenGL instanced cubes as the live reconstruction. They give us controlled geometry, composition and repeatable stimuli without making the entire demo dependent on noisy monocular CV.

Preset modes currently include a small moving orange **stimulus probe** so the visual-pathway/agent loop can be tested repeatably. It is a controlled experiment marker, not a claim about natural fly behaviour.

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

### M2 — preset worlds + fly agent + visual stimulus 🚧

The discarded screen-space “compound eye” filter has been removed. The fly now exists as an agent inside the same voxel world that the researcher inspects.

```text
scene stimulus / live CV motion
  ↓
3D target
  ↓
retina → lamina → medulla → lobula
  ↓
prototype steering response
  ↓
fly moves inside the OpenGL voxel scene
```

The four named stages are real major parts of the fly visual system, but the current scalar transfer functions and steering policy are deliberately **engineering placeholders**, not a claim of biologically faithful neural simulation. M3 will replace this proxy with connectivity grounded in a real Drosophila visual-neural map.

Four small activity nodes float above the fly in the order `retina → lamina → medulla → lobula`; their activation follows the stimulus. The fly body, neural nodes, preset worlds and live reconstruction all use the same OpenGL instanced-cube renderer.

## Locked dashboard target

The final prototype UI is organised around **cause → processing → neural activity → behaviour**, rather than nine unrelated visual effects.

```text
┌───────────────┬───────────────┬───────────────┬───────────────┐
│ 1 SOURCE RGB  │ 2 DEPTH       │ 3 SEGMENTATION│ 4 MOTION      │
│ scene/camera  │ geometry cue  │ semantic cue  │ flow/looming  │
├───────────────────────────────────────────┬───────────────────┤
│                                           │ 6 NEURAL MAP      │
│                                           │ active circuits   │
│          5 MAIN VOXEL WORLD               ├───────────────────┤
│      researcher + fly agent               │ 7 NEURAL RENDER   │
│                                           │ learned view      │
│                                           ├───────────────────┤
│                                           │ 8 BEHAVIOUR       │
│                                           │ heading/path      │
├───────────────────────────────────────────┴───────────────────┤
│ 9 CAUSAL TRACE / TIMELINE                                     │
│ stimulus → feature → neural population → motor response        │
└───────────────────────────────────────────────────────────────┘
```

The main voxel world should remain roughly **55–60% of the usable display area**. Supporting panels reuse CV buffers, render targets and intermediate state rather than rendering nine heavyweight independent worlds.

The intended nine realtime surfaces are:

1. source RGB / selected scene input
2. depth
3. segmentation
4. motion / looming / visual stimulus features
5. main OpenGL voxel world
6. fly neural map activity
7. bounded neural-rendering preview
8. behaviour output / trajectory
9. causal trace timeline

There is deliberately **no fake “what the fly literally sees” honeycomb panel** in the locked design. If a perceptual view returns later, it must correspond to a scientifically meaningful computation rather than a decorative filter.

## Planned full pipeline

```text
SCENE SOURCE
(preset world or live camera)
    ↓
REAL-TIME CV / CONTROLLED STIMULUS
(depth / segmentation / optical flow / looming)
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
(bounded learned rendering experiment)
```

## Controls

- `1` — Office / Studio
- `2` — Kitchen / Fruit Table
- `3` — Live Camera Reconstruction
- `Esc` — quit
- `Space` — pause/resume updates
- `A` — enable/disable fly steering while keeping neural activity visible
- `D` — toggle monocular depth vs brightness fallback in live mode
- `R` — reset researcher camera orbit
- Drag with **left mouse** — orbit
- Mouse wheel — zoom

## Dependencies

- C++20 compiler
- OpenGL 3.3+
- CMake 3.24+ (Visual Studio 2026 users need a recent CMake that knows the VS 18 generator)
- webcam for live mode
- [vcpkg](https://github.com/microsoft/vcpkg)

The manifest installs GLEW, GLFW, GLM and OpenCV including the DNN module.

## Get the depth model

The model weights are deliberately **not committed** to this repository.

```powershell
.\scripts\download_depth_model.ps1
```

This downloads MiDaS v2.1 Small to `models/model-small.onnx`. If the model is absent or depth inference fails, live mode stays runnable using the original image-intensity fallback.

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
- [ ] **M2:** preset scenes + fly agent + repeatable visual stimulus (implemented; awaiting local validation)
- [ ] **M3:** real Drosophila visual-pathway map + live activation
- [ ] **M4:** nine-surface researcher dashboard + interaction / inspection modes
- [ ] **M5:** bounded neural-rendering experiment
- [ ] **M6:** public demo video + technical write-up

## Design rule

The neural component must **augment rather than replace the OpenGL renderer**. Geometry, camera, buffers, transforms, instancing, shading, render targets and interaction remain explicit graphics work.
