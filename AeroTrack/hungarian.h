#pragma once
#include <vector>
#include <utility>

// Basic Hungarian/Munkres solver for dense cost matrices (min-cost assignment)
std::vector<std::pair<int, int>> hungarianAssign(const std::vector<std::vector<float>> &costMatrix, float maxCost);
