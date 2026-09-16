#include "preset_scenes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr float kStep = 0.16f;
constexpr float kScale = 0.077f;

float hash01(glm::vec3 p) {
    const int xi = static_cast<int>(std::round(p.x / kStep));
    const int yi = static_cast<int>(std::round(p.y / kStep));
    const int zi = static_cast<int>(std::round(p.z / kStep));
    std::uint32_t h = static_cast<std::uint32_t>(xi * 73856093) ^
                      static_cast<std::uint32_t>(yi * 19349663) ^
                      static_cast<std::uint32_t>(zi * 83492791);
    h ^= h >> 13;
    h *= 1274126177u;
    return static_cast<float>(h & 0xffffu) / 65535.0f;
}

glm::vec3 varied(glm::vec3 colour, glm::vec3 p, float amount = 0.035f) {
    const float n = (hash01(p) - 0.5f) * 2.0f;
    return glm::clamp(colour * (1.0f + n * amount), glm::vec3(0.0f), glm::vec3(1.0f));
}

void addVoxel(PresetScene& scene, glm::vec3 p, glm::vec3 colour, float scale = kScale, float variation = 0.035f) {
    scene.voxels.push_back({p, varied(colour, p, variation), scale});
}

void addPanelXZ(PresetScene& scene, glm::vec3 centre, int nx, int nz, glm::vec3 colour, float variation = 0.035f) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            const float x = centre.x + (ix - (nx - 1) * 0.5f) * kStep;
            const float z = centre.z + (iz - (nz - 1) * 0.5f) * kStep;
            addVoxel(scene, {x, centre.y, z}, colour, kScale, variation);
        }
    }
}

void addPanelXY(PresetScene& scene, glm::vec3 centre, int nx, int ny, glm::vec3 colour, float variation = 0.035f) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            const float x = centre.x + (ix - (nx - 1) * 0.5f) * kStep;
            const float y = centre.y + (iy - (ny - 1) * 0.5f) * kStep;
            addVoxel(scene, {x, y, centre.z}, colour, kScale, variation);
        }
    }
}

void addPanelYZ(PresetScene& scene, glm::vec3 centre, int ny, int nz, glm::vec3 colour, float variation = 0.035f) {
    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            const float y = centre.y + (iy - (ny - 1) * 0.5f) * kStep;
            const float z = centre.z + (iz - (nz - 1) * 0.5f) * kStep;
            addVoxel(scene, {centre.x, y, z}, colour, kScale, variation);
        }
    }
}

void addBox(PresetScene& scene, glm::vec3 centre, int nx, int ny, int nz, glm::vec3 colour, float variation = 0.025f) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int iz = 0; iz < nz; ++iz) {
                const float x = centre.x + (ix - (nx - 1) * 0.5f) * kStep;
                const float y = centre.y + (iy - (ny - 1) * 0.5f) * kStep;
                const float z = centre.z + (iz - (nz - 1) * 0.5f) * kStep;
                addVoxel(scene, {x, y, z}, colour, kScale, variation);
            }
        }
    }
}

void addSphere(PresetScene& scene, glm::vec3 centre, float radius, glm::vec3 colour, float variation = 0.03f) {
    const int r = std::max(1, static_cast<int>(std::ceil(radius / kStep)));
    for (int x = -r; x <= r; ++x) {
        for (int y = -r; y <= r; ++y) {
            for (int z = -r; z <= r; ++z) {
                const glm::vec3 offset(x * kStep, y * kStep, z * kStep);
                if (glm::dot(offset, offset) <= radius * radius) {
                    addVoxel(scene, centre + offset, colour, kScale, variation);
                }
            }
        }
    }
}

void addShadowPatch(PresetScene& scene, glm::vec3 centre, int nx, int nz, glm::vec3 colour) {
    // Slightly above the floor: a deliberately simple baked contact-shadow proxy.
    addPanelXZ(scene, centre, nx, nz, colour, 0.015f);
}

void addPlant(PresetScene& scene, glm::vec3 base, float scale = 1.0f) {
    const glm::vec3 pot(0.66f, 0.47f, 0.31f);
    const glm::vec3 rim(0.76f, 0.58f, 0.39f);
    const glm::vec3 soil(0.20f, 0.14f, 0.09f);
    const glm::vec3 stem(0.17f, 0.31f, 0.13f);
    const glm::vec3 leafA(0.20f, 0.49f, 0.17f);
    const glm::vec3 leafB(0.31f, 0.62f, 0.23f);

    addBox(scene, base, 3, 3, 3, pot);
    addBox(scene, base + glm::vec3(0.0f, 0.24f, 0.0f) * scale, 4, 1, 4, rim);
    addBox(scene, base + glm::vec3(0.0f, 0.31f, 0.0f) * scale, 3, 1, 3, soil, 0.08f);
    addBox(scene, base + glm::vec3(0.0f, 0.65f, 0.0f) * scale, 1, 6, 1, stem);

    addBox(scene, base + glm::vec3(-0.32f, 0.73f, 0.02f) * scale, 4, 1, 2, leafA);
    addBox(scene, base + glm::vec3(0.33f, 0.92f, -0.04f) * scale, 4, 1, 2, leafB);
    addBox(scene, base + glm::vec3(-0.10f, 1.12f, 0.04f) * scale, 3, 1, 4, leafB);
    addBox(scene, base + glm::vec3(0.18f, 1.30f, -0.02f) * scale, 2, 1, 3, leafA);
}

void addChair(PresetScene& scene, glm::vec3 seatCentre, glm::vec3 colour) {
    const glm::vec3 frame = colour * 0.67f;
    const glm::vec3 cushion = glm::clamp(colour * 1.10f, glm::vec3(0.0f), glm::vec3(1.0f));

    addBox(scene, seatCentre, 6, 1, 6, cushion);
    addBox(scene, seatCentre + glm::vec3(0.0f, 0.66f, 0.42f), 6, 8, 1, colour);
    addBox(scene, seatCentre + glm::vec3(0.0f, 0.66f, 0.34f), 4, 5, 1, cushion);
    for (float sx : {-0.34f, 0.34f}) {
        for (float sz : {-0.34f, 0.34f}) {
            addBox(scene, seatCentre + glm::vec3(sx, -0.58f, sz), 1, 7, 1, frame);
        }
    }
}

void addWindow(PresetScene& scene, glm::vec3 centre, int nx, int ny) {
    const glm::vec3 skyTop(0.45f, 0.72f, 0.94f);
    const glm::vec3 skyBottom(0.75f, 0.88f, 0.98f);
    const glm::vec3 frame(0.88f, 0.86f, 0.78f);

    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            const float x = centre.x + (ix - (nx - 1) * 0.5f) * kStep;
            const float y = centre.y + (iy - (ny - 1) * 0.5f) * kStep;
            const float t = static_cast<float>(iy) / static_cast<float>(std::max(ny - 1, 1));
            const glm::vec3 c = skyBottom * (1.0f - t) + skyTop * t;
            addVoxel(scene, {x, y, centre.z}, c, kScale, 0.015f);
        }
    }

    // Window frame and mullions.
    addBox(scene, centre + glm::vec3(0.0f, (ny * kStep) * 0.52f, 0.04f), nx + 2, 1, 1, frame);
    addBox(scene, centre - glm::vec3(0.0f, (ny * kStep) * 0.52f, -0.04f), nx + 2, 1, 1, frame);
    addBox(scene, centre - glm::vec3((nx * kStep) * 0.52f, 0.0f, -0.04f), 1, ny + 2, 1, frame);
    addBox(scene, centre + glm::vec3((nx * kStep) * 0.52f, 0.0f, 0.04f), 1, ny + 2, 1, frame);
    addBox(scene, centre + glm::vec3(0.0f, 0.0f, 0.05f), 1, ny, 1, frame * 0.96f);
    addBox(scene, centre + glm::vec3(0.0f, 0.0f, 0.05f), nx, 1, 1, frame * 0.96f);

    // Simplified skyline beyond the glass to avoid a flat terminal plane.
    const glm::vec3 cityA(0.34f, 0.43f, 0.49f);
    const glm::vec3 cityB(0.42f, 0.51f, 0.56f);
    for (int i = -5; i <= 5; ++i) {
        const int height = 2 + (std::abs(i * 7 + 3) % 5);
        addBox(scene,
               centre + glm::vec3(i * 0.27f, -0.62f + height * 0.08f, -0.24f),
               1, height, 2,
               (i % 2 == 0) ? cityA : cityB,
               0.025f);
    }
}

void addSunPatch(PresetScene& scene, glm::vec3 centre, int nx, int nz) {
    const glm::vec3 sunlight(0.91f, 0.70f, 0.39f);
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            // Broken-up patch avoids reading as a literal yellow rectangle.
            const glm::vec3 p{
                centre.x + (ix - (nx - 1) * 0.5f) * kStep,
                centre.y,
                centre.z + (iz - (nz - 1) * 0.5f) * kStep
            };
            if (hash01(p + glm::vec3(2.4f, 0.0f, 1.7f)) > 0.17f) {
                addVoxel(scene, p, sunlight, 0.078f, 0.10f);
            }
        }
    }
}

} // namespace

PresetScene makeOfficeScene() {
    PresetScene scene;
    scene.name = "Office / Studio";
    scene.flyStart = {0.15f, -0.35f, 1.65f};
    scene.focusTarget = {0.30f, -0.30f, -0.85f};
    scene.stimulusAnchor = {1.30f, 0.25f, -1.10f};

    const glm::vec3 floor(0.66f, 0.53f, 0.38f);
    const glm::vec3 wall(0.75f, 0.73f, 0.68f);
    const glm::vec3 darkWall(0.42f, 0.45f, 0.47f);
    const glm::vec3 desk(0.43f, 0.25f, 0.13f);
    const glm::vec3 metal(0.15f, 0.17f, 0.18f);
    const glm::vec3 screen(0.040f, 0.060f, 0.075f);

    addPanelXZ(scene, {0.0f, -1.62f, 0.0f}, 46, 34, floor, 0.055f);
    addPanelXY(scene, {0.0f, 0.25f, -2.70f}, 46, 24, wall, 0.025f);
    addPanelYZ(scene, {-3.58f, 0.25f, 0.0f}, 24, 34, darkWall, 0.035f);

    // Architectural trim gives the room a clearer silhouette.
    addBox(scene, {0.0f, -1.43f, -2.58f}, 46, 1, 1, wall * 0.78f);
    addBox(scene, {-3.45f, -1.43f, 0.0f}, 1, 1, 34, darkWall * 0.72f);

    addWindow(scene, {2.10f, 0.70f, -2.67f}, 14, 12);
    addBox(scene, {2.10f, -0.32f, -2.52f}, 17, 2, 2, glm::vec3(0.82f, 0.79f, 0.70f));

    // Desk with a slightly lighter work surface and darker frame.
    addBox(scene, {0.25f, -0.39f, -1.35f}, 22, 2, 7, desk);
    addBox(scene, {0.25f, -0.24f, -1.35f}, 22, 1, 7, desk * 1.16f);
    for (float x : {-1.20f, 1.65f}) addBox(scene, {x, -1.00f, -1.35f}, 2, 7, 2, metal);

    // Monitor, stand and a cool screen reflection.
    addBox(scene, {0.15f, 0.54f, -1.52f}, 11, 8, 1, glm::vec3(0.08f, 0.09f, 0.10f));
    addBox(scene, {0.15f, 0.54f, -1.43f}, 9, 6, 1, screen * 1.55f, 0.01f);
    addBox(scene, {0.15f, 0.02f, -1.48f}, 2, 4, 2, metal);
    addBox(scene, {0.15f, -0.18f, -1.43f}, 5, 1, 3, metal);

    // Keyboard, mouse, books, mug and desk lamp.
    addBox(scene, {0.10f, -0.15f, -0.91f}, 8, 1, 3, glm::vec3(0.72f, 0.73f, 0.71f));
    addBox(scene, {0.86f, -0.15f, -0.87f}, 2, 1, 2, glm::vec3(0.24f, 0.25f, 0.24f));
    addBox(scene, {1.30f, -0.04f, -0.91f}, 3, 4, 3, glm::vec3(0.67f, 0.15f, 0.10f));
    addBox(scene, {-1.08f, -0.12f, -0.92f}, 5, 1, 3, glm::vec3(0.19f, 0.31f, 0.53f));
    addBox(scene, {-1.08f, -0.03f, -0.92f}, 4, 1, 3, glm::vec3(0.70f, 0.53f, 0.23f));
    addBox(scene, {1.62f, 0.12f, -1.15f}, 1, 5, 1, glm::vec3(0.22f, 0.22f, 0.19f));
    addBox(scene, {1.48f, 0.48f, -1.15f}, 4, 1, 3, glm::vec3(0.82f, 0.68f, 0.38f));

    addShadowPatch(scene, {-0.32f, -1.535f, 0.45f}, 9, 9, glm::vec3(0.30f, 0.25f, 0.21f));
    addChair(scene, {-0.35f, -0.62f, 0.45f}, glm::vec3(0.16f, 0.18f, 0.19f));

    addShadowPatch(scene, {2.45f, -1.535f, -0.65f}, 6, 6, glm::vec3(0.34f, 0.28f, 0.22f));
    addPlant(scene, {2.45f, -1.34f, -0.65f});

    // Shelving and small colour accents create depth behind the desk.
    const glm::vec3 shelf(0.36f, 0.23f, 0.14f);
    addBox(scene, {-2.42f, -0.15f, -2.34f}, 8, 1, 4, shelf);
    addBox(scene, {-2.42f, 0.82f, -2.34f}, 8, 1, 4, shelf);
    addBox(scene, {-2.94f, 0.35f, -2.34f}, 1, 7, 4, shelf * 0.80f);
    addBox(scene, {-1.90f, 0.35f, -2.34f}, 1, 7, 4, shelf * 0.80f);
    addBox(scene, {-2.58f, 0.05f, -2.05f}, 2, 4, 2, glm::vec3(0.57f, 0.24f, 0.17f));
    addBox(scene, {-2.28f, 0.04f, -2.05f}, 2, 4, 2, glm::vec3(0.18f, 0.35f, 0.49f));
    addBox(scene, {-2.42f, 0.98f, -2.05f}, 3, 2, 2, glm::vec3(0.58f, 0.54f, 0.28f));

    // Baked warm daylight patch from the right-side window.
    addSunPatch(scene, {1.18f, -1.535f, 0.70f}, 12, 10);

    return scene;
}

PresetScene makeKitchenScene() {
    PresetScene scene;
    scene.name = "Kitchen / Fruit Table";
    scene.flyStart = {0.15f, -0.28f, 1.70f};
    scene.focusTarget = {0.10f, -0.38f, -0.35f};
    scene.stimulusAnchor = {0.20f, 0.00f, -0.35f};

    const glm::vec3 floor(0.69f, 0.58f, 0.42f);
    const glm::vec3 wall(0.82f, 0.79f, 0.71f);
    const glm::vec3 table(0.43f, 0.25f, 0.13f);
    const glm::vec3 cabinet(0.55f, 0.49f, 0.39f);
    const glm::vec3 counter(0.23f, 0.24f, 0.23f);

    addPanelXZ(scene, {0.0f, -1.62f, 0.0f}, 46, 34, floor, 0.055f);
    addPanelXY(scene, {0.0f, 0.25f, -2.70f}, 46, 24, wall, 0.025f);
    addPanelYZ(scene, {-3.58f, 0.25f, 0.0f}, 24, 34, wall * 0.84f, 0.03f);
    addBox(scene, {0.0f, -1.43f, -2.58f}, 46, 1, 1, wall * 0.78f);

    addWindow(scene, {2.12f, 0.94f, -2.67f}, 14, 10);
    addBox(scene, {2.12f, 0.06f, -2.53f}, 17, 2, 2, glm::vec3(0.84f, 0.81f, 0.72f));

    // Back cabinetry with visible doors, handles, backsplash and worktop.
    addBox(scene, {-1.46f, -0.74f, -2.31f}, 17, 9, 4, cabinet);
    addBox(scene, {1.46f, -0.74f, -2.31f}, 17, 9, 4, cabinet * 0.96f);
    addBox(scene, {0.0f, -0.02f, -2.14f}, 36, 2, 6, counter);
    addPanelXY(scene, {0.0f, 0.43f, -2.56f}, 34, 4, glm::vec3(0.71f, 0.72f, 0.68f), 0.025f);
    for (float x : {-2.45f, -1.55f, -0.65f, 0.65f, 1.55f, 2.45f}) {
        addBox(scene, {x, -0.74f, -2.05f}, 1, 7, 1, cabinet * 0.72f);
        addBox(scene, {x + 0.12f, -0.66f, -1.98f}, 1, 2, 1, glm::vec3(0.18f, 0.18f, 0.16f));
    }

    // Dining table: warmer top surface and grounded legs.
    addShadowPatch(scene, {0.0f, -1.535f, -0.10f}, 25, 18, glm::vec3(0.33f, 0.27f, 0.21f));
    addBox(scene, {0.0f, -0.49f, -0.20f}, 22, 2, 14, table);
    addBox(scene, {0.0f, -0.33f, -0.20f}, 22, 1, 14, table * 1.16f);
    for (float x : {-1.45f, 1.45f}) {
        for (float z : {-0.95f, 0.55f}) addBox(scene, {x, -1.05f, z}, 2, 7, 2, table * 0.76f);
    }

    addChair(scene, {-2.05f, -0.70f, 0.65f}, glm::vec3(0.23f, 0.20f, 0.18f));
    addChair(scene, {2.05f, -0.70f, 0.65f}, glm::vec3(0.23f, 0.20f, 0.18f));

    // Fruit bowl and distinct fruit silhouettes.
    addBox(scene, {0.18f, -0.18f, -0.35f}, 9, 1, 7, glm::vec3(0.58f, 0.43f, 0.25f));
    addBox(scene, {0.18f, -0.08f, -0.35f}, 7, 1, 5, glm::vec3(0.70f, 0.53f, 0.30f));
    addSphere(scene, {-0.18f, 0.10f, -0.34f}, 0.30f, glm::vec3(0.80f, 0.10f, 0.06f));
    addVoxel(scene, {-0.18f, 0.42f, -0.34f}, glm::vec3(0.19f, 0.29f, 0.10f), kScale, 0.01f);
    addSphere(scene, {0.32f, 0.11f, -0.22f}, 0.28f, glm::vec3(0.94f, 0.47f, 0.04f));
    addSphere(scene, {0.12f, 0.29f, -0.52f}, 0.23f, glm::vec3(0.48f, 0.68f, 0.15f));
    addBox(scene, {0.64f, 0.26f, -0.44f}, 6, 1, 2, glm::vec3(0.94f, 0.76f, 0.10f));
    addBox(scene, {0.80f, 0.34f, -0.44f}, 3, 1, 2, glm::vec3(0.84f, 0.68f, 0.08f));

    // Mug, cutting board, herbs and pendant lamp.
    addBox(scene, {-1.05f, -0.14f, -0.05f}, 3, 4, 3, glm::vec3(0.10f, 0.37f, 0.55f));
    addBox(scene, {-1.42f, -0.25f, -0.42f}, 5, 1, 7, glm::vec3(0.65f, 0.46f, 0.26f));
    addShadowPatch(scene, {2.55f, -1.535f, -1.00f}, 6, 6, glm::vec3(0.34f, 0.28f, 0.22f));
    addPlant(scene, {2.55f, -1.34f, -1.00f});

    addBox(scene, {0.0f, 2.13f, -0.20f}, 7, 2, 7, glm::vec3(0.82f, 0.66f, 0.31f));
    addBox(scene, {0.0f, 2.51f, -0.20f}, 1, 4, 1, glm::vec3(0.21f, 0.20f, 0.18f));
    addBox(scene, {0.0f, 1.95f, -0.20f}, 4, 1, 4, glm::vec3(0.96f, 0.79f, 0.42f), 0.01f);

    // Warm window light across table/floor for a more intentional hero composition.
    addSunPatch(scene, {1.15f, -1.535f, 0.62f}, 13, 11);
    addSunPatch(scene, {1.05f, -0.255f, -0.15f}, 8, 7);

    return scene;
}
