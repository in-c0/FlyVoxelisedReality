#pragma once

#include <glm/glm.hpp>

#include <string>
#include <vector>

struct SceneVoxel {
    glm::vec3 position{0.0f};
    glm::vec3 colour{1.0f};
    float scale = 0.08f;
};

struct PresetScene {
    std::string name;
    std::vector<SceneVoxel> voxels;
    glm::vec3 flyStart{0.0f, 0.0f, 2.2f};
    glm::vec3 focusTarget{0.0f};
    glm::vec3 stimulusAnchor{0.0f};
};

PresetScene makeOfficeScene();
PresetScene makeKitchenScene();
