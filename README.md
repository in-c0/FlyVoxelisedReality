# FlyVoxelisedReality

**Reality Voxelised Through a Fly** — an experimental real-time graphics prototype exploring a pipeline from live computer vision to an OpenGL voxel world, fly-inspired perception, mapped Drosophila neural activity, and learned/neural appearance.

> This repository is an independent technical prototype used to validate the concept before any course group project is scoped. It is intentionally narrower than a complete game.

## v0 goal

Prove the first risky link in the chain with a runnable desktop application:

```text
webcam
  ↓
OpenCV preprocessing + optical flow
  ↓
coarse live voxel field
  ↓
OpenGL instanced rendering
```

The first version uses image intensity as a deliberately simple pseudo-depth signal so the camera-to-voxel path is testable without downloading a depth model. True monocular/RGB-D depth, fly-vision transforms, connectome activity, and neural rendering are staged behind later milestones.

## Planned pipeline

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

## v0 controls

- `Esc` — quit
- `Space` — pause/resume camera-driven voxel updates
- `R` — reset camera orbit
- Drag with **left mouse** — orbit
- Mouse wheel — zoom

## Build

Requirements:

- CMake 3.24+
- C++20 compiler
- OpenGL 3.3+
- a webcam
- [vcpkg](https://github.com/microsoft/vcpkg)

Install dependencies through the manifest:

```bash
vcpkg install
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
```

Then run:

```bash
./build/FlyVoxelisedReality
```

On Visual Studio multi-config generators the executable is usually under `build/Release/`.

## Milestones

- [x] Repository + architecture scaffold
- [ ] **M0:** webcam → OpenCV → live OpenGL voxel field
- [ ] **M1:** true depth/segmentation input
- [ ] **M2:** fly-inspired compound-eye / motion perception
- [ ] **M3:** real Drosophila visual-pathway map + live activation
- [ ] **M4:** researcher ↔ fly dual perspective
- [ ] **M5:** bounded neural-rendering experiment
- [ ] **M6:** public demo video + technical write-up

## Design rule

The neural component must **augment rather than replace the OpenGL renderer**. Geometry, camera, buffers, transforms, instancing, shading, render targets, and interaction remain explicit graphics work.
