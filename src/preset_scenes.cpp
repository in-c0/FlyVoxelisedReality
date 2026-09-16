#include "preset_scenes.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kStep = 0.16f;
constexpr float kScale = 0.077f;

void addVoxel(PresetScene& scene, glm::vec3 p, glm::vec3 colour, float scale = kScale) {
    scene.voxels.push_back({p, colour, scale});
}

void addPanelXZ(PresetScene& scene, glm::vec3 centre, int nx, int nz, glm::vec3 colour) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iz = 0; iz < nz; ++iz) {
            const float x = centre.x + (ix - (nx - 1) * 0.5f) * kStep;
            const float z = centre.z + (iz - (nz - 1) * 0.5f) * kStep;
            addVoxel(scene, {x, centre.y, z}, colour);
        }
    }
}

void addPanelXY(PresetScene& scene, glm::vec3 centre, int nx, int ny, glm::vec3 colour) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            const float x = centre.x + (ix - (nx - 1) * 0.5f) * kStep;
            const float y = centre.y + (iy - (ny - 1) * 0.5f) * kStep;
            addVoxel(scene, {x, y, centre.z}, colour);
        }
    }
}

void addPanelYZ(PresetScene& scene, glm::vec3 centre, int ny, int nz, glm::vec3 colour) {
    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            const float y = centre.y + (iy - (ny - 1) * 0.5f) * kStep;
            const float z = centre.z + (iz - (nz - 1) * 0.5f) * kStep;
            addVoxel(scene, {centre.x, y, z}, colour);
        }
    }
}

void addBox(PresetScene& scene, glm::vec3 centre, int nx, int ny, int nz, glm::vec3 colour) {
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int iz = 0; iz < nz; ++iz) {
                const float x = centre.x + (ix - (nx - 1) * 0.5f) * kStep;
                const float y = centre.y + (iy - (ny - 1) * 0.5f) * kStep;
                const float z = centre.z + (iz - (nz - 1) * 0.5f) * kStep;
                addVoxel(scene, {x, y, z}, colour);
            }
        }
    }
}

void addSphere(PresetScene& scene, glm::vec3 centre, float radius, glm::vec3 colour) {
    const int r = std::max(1, static_cast<int>(std::ceil(radius / kStep)));
    for (int x = -r; x <= r; ++x) {
        for (int y = -r; y <= r; ++y) {
            for (int z = -r; z <= r; ++z) {
                glm::vec3 offset(x * kStep, y * kStep, z * kStep);
                if (glm::dot(offset, offset) <= radius * radius) addVoxel(scene, centre + offset, colour);
            }
        }
    }
}

void addPlant(PresetScene& scene, glm::vec3 base) {
    const glm::vec3 pot(0.72f, 0.55f, 0.38f);
    const glm::vec3 stem(0.20f, 0.36f, 0.16f);
    const glm::vec3 leafA(0.24f, 0.54f, 0.20f);
    const glm::vec3 leafB(0.36f, 0.68f, 0.28f);
    addBox(scene, base, 3, 3, 3, pot);
    addBox(scene, base + glm::vec3(0.0f, 0.48f, 0.0f), 1, 5, 1, stem);
    addBox(scene, base + glm::vec3(-0.30f, 0.72f, 0.0f), 4, 1, 2, leafA);
    addBox(scene, base + glm::vec3(0.28f, 0.92f, 0.02f), 4, 1, 2, leafB);
    addBox(scene, base + glm::vec3(-0.08f, 1.15f, -0.04f), 3, 1, 3, leafB);
}

void addChair(PresetScene& scene, glm::vec3 seatCentre, glm::vec3 colour) {
    addBox(scene, seatCentre, 6, 1, 6, colour);
    addBox(scene, seatCentre + glm::vec3(0.0f, 0.65f, 0.42f), 6, 8, 1, colour);
    for (float sx : {-0.34f, 0.34f}) {
        for (float sz : {-0.34f, 0.34f}) {
            addBox(scene, seatCentre + glm::vec3(sx, -0.58f, sz), 1, 7, 1, colour * 0.82f);
        }
    }
}

} // namespace

PresetScene makeOfficeScene() {
    PresetScene scene;
    scene.name = "Office / Studio";
    scene.flyStart = {0.0f, 0.2f, 2.25f};
    scene.focusTarget = {0.0f, 0.0f, -0.25f};
    scene.stimulusAnchor = {1.35f, 0.55f, -1.25f};

    const glm::vec3 floor(0.69f, 0.58f, 0.43f);
    const glm::vec3 wall(0.78f, 0.76f, 0.71f);
    const glm::vec3 darkWall(0.54f, 0.56f, 0.57f);
    const glm::vec3 desk(0.47f, 0.30f, 0.18f);
    const glm::vec3 metal(0.20f, 0.22f, 0.24f);
    const glm::vec3 screen(0.055f, 0.075f, 0.090f);
    const glm::vec3 window(0.68f, 0.84f, 0.96f);

    addPanelXZ(scene, {0.0f, -1.62f, 0.0f}, 46, 34, floor);
    addPanelXY(scene, {0.0f, 0.25f, -2.70f}, 46, 24, wall);
    addPanelYZ(scene, {-3.58f, 0.25f, 0.0f}, 24, 34, darkWall);

    // Window recess and bright panes.
    addBox(scene, {2.18f, 0.60f, -2.55f}, 15, 1, 1, glm::vec3(0.92f, 0.91f, 0.84f));
    addPanelXY(scene, {2.18f, 0.72f, -2.66f}, 14, 12, window);
    addBox(scene, {2.18f, 0.72f, -2.59f}, 1, 12, 1, glm::vec3(0.86f));

    // Desk and monitor.
    addBox(scene, {0.25f, -0.38f, -1.35f}, 22, 2, 7, desk);
    for (float x : {-1.2f, 1.65f}) addBox(scene, {x, -1.0f, -1.35f}, 2, 7, 2, metal);
    addBox(scene, {0.15f, 0.54f, -1.52f}, 10, 7, 1, screen);
    addBox(scene, {0.15f, 0.02f, -1.48f}, 2, 4, 2, metal);
    addBox(scene, {0.15f, -0.18f, -1.43f}, 5, 1, 3, metal);

    // Keyboard, mug, small books.
    addBox(scene, {0.15f, -0.16f, -0.92f}, 8, 1, 3, glm::vec3(0.76f, 0.77f, 0.75f));
    addBox(scene, {1.25f, -0.05f, -0.92f}, 3, 3, 3, glm::vec3(0.64f, 0.17f, 0.11f));
    addBox(scene, {-1.15f, -0.08f, -0.92f}, 5, 1, 3, glm::vec3(0.20f, 0.34f, 0.55f));

    addChair(scene, {-0.35f, -0.62f, 0.45f}, glm::vec3(0.19f, 0.21f, 0.22f));
    addPlant(scene, {2.45f, -1.34f, -0.65f});

    // Shelf / visual depth cue.
    addBox(scene, {-2.40f, -0.15f, -2.35f}, 8, 1, 4, glm::vec3(0.42f, 0.28f, 0.19f));
    addBox(scene, {-2.40f, 0.82f, -2.35f}, 8, 1, 4, glm::vec3(0.42f, 0.28f, 0.19f));
    addBox(scene, {-2.92f, 0.35f, -2.35f}, 1, 7, 4, glm::vec3(0.36f, 0.24f, 0.17f));
    addBox(scene, {-1.88f, 0.35f, -2.35f}, 1, 7, 4, glm::vec3(0.36f, 0.24f, 0.17f));

    return scene;
}

PresetScene makeKitchenScene() {
    PresetScene scene;
    scene.name = "Kitchen / Fruit Table";
    scene.flyStart = {0.0f, 0.35f, 2.35f};
    scene.focusTarget = {0.0f, -0.25f, -0.20f};
    scene.stimulusAnchor = {0.20f, -0.05f, -0.35f};

    const glm::vec3 floor(0.72f, 0.62f, 0.47f);
    const glm::vec3 wall(0.84f, 0.81f, 0.74f);
    const glm::vec3 table(0.48f, 0.30f, 0.17f);
    const glm::vec3 cabinet(0.62f, 0.56f, 0.45f);
    const glm::vec3 window(0.65f, 0.84f, 0.97f);

    addPanelXZ(scene, {0.0f, -1.62f, 0.0f}, 46, 34, floor);
    addPanelXY(scene, {0.0f, 0.25f, -2.70f}, 46, 24, wall);
    addPanelYZ(scene, {-3.58f, 0.25f, 0.0f}, 24, 34, glm::vec3(0.73f, 0.70f, 0.64f));

    // Back cabinets and worktop.
    addBox(scene, {-1.45f, -0.72f, -2.32f}, 17, 9, 4, cabinet);
    addBox(scene, {1.45f, -0.72f, -2.32f}, 17, 9, 4, cabinet * 0.96f);
    addBox(scene, {0.0f, -0.02f, -2.15f}, 36, 2, 6, glm::vec3(0.28f, 0.29f, 0.28f));

    addPanelXY(scene, {2.10f, 0.92f, -2.66f}, 14, 10, window);

    // Dining table.
    addBox(scene, {0.0f, -0.48f, -0.20f}, 22, 2, 14, table);
    for (float x : {-1.45f, 1.45f}) {
        for (float z : {-0.95f, 0.55f}) addBox(scene, {x, -1.05f, z}, 2, 7, 2, table * 0.82f);
    }

    addChair(scene, {-2.05f, -0.70f, 0.65f}, glm::vec3(0.28f, 0.24f, 0.21f));
    addChair(scene, {2.05f, -0.70f, 0.65f}, glm::vec3(0.28f, 0.24f, 0.21f));

    // Fruit bowl and fruit cluster.
    addBox(scene, {0.20f, -0.20f, -0.35f}, 9, 1, 7, glm::vec3(0.62f, 0.47f, 0.29f));
    addSphere(scene, {-0.12f, 0.08f, -0.35f}, 0.31f, glm::vec3(0.82f, 0.13f, 0.08f));
    addSphere(scene, {0.35f, 0.10f, -0.25f}, 0.29f, glm::vec3(0.95f, 0.55f, 0.05f));
    addSphere(scene, {0.14f, 0.28f, -0.52f}, 0.25f, glm::vec3(0.56f, 0.72f, 0.18f));
    addBox(scene, {0.60f, 0.28f, -0.45f}, 5, 1, 2, glm::vec3(0.94f, 0.78f, 0.12f));

    // Mug, herbs, pendant light.
    addBox(scene, {-1.05f, -0.18f, -0.05f}, 3, 4, 3, glm::vec3(0.13f, 0.40f, 0.58f));
    addPlant(scene, {2.55f, -1.34f, -1.00f});
    addBox(scene, {0.0f, 2.15f, -0.20f}, 5, 2, 5, glm::vec3(0.88f, 0.73f, 0.38f));
    addBox(scene, {0.0f, 2.52f, -0.20f}, 1, 4, 1, glm::vec3(0.26f, 0.24f, 0.21f));

    return scene;
}
