#include "hungarian.h"
#include <algorithm>

std::vector<std::pair<int, int>> hungarianAssign(const std::vector<std::vector<float>> &costMatrix, float maxCost)
{
    if (costMatrix.empty())
        return {};

    constexpr float INF = 1e6f;
    const int n = static_cast<int>(costMatrix.size());
    const int m = static_cast<int>(costMatrix[0].size());
    const int dim = std::max(n, m);

    std::vector<std::vector<float>> cost(dim, std::vector<float>(dim, INF));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            cost[i][j] = costMatrix[i][j];

    std::vector<float> u(dim + 1), v(dim + 1);
    std::vector<int> p(dim + 1), way(dim + 1);

    for (int i = 1; i <= dim; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(dim + 1, INF);
        std::vector<char> used(dim + 1, false);

        do
        {
            used[j0] = true;
            const int i0 = p[j0];
            int j1 = 0;
            float delta = INF;

            for (int j = 1; j <= dim; ++j)
            {
                if (used[j])
                    continue;

                const float cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for (int j = 0; j <= dim; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do
        {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<std::pair<int, int>> assignment;
    assignment.reserve(dim);
    for (int j = 1; j <= dim; ++j)
    {
        if (p[j] == 0 || p[j] > n || j > m)
            continue;
        if (cost[p[j] - 1][j - 1] < maxCost)
            assignment.emplace_back(p[j] - 1, j - 1);
    }
    return assignment;
}
