#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

#include "Configs.h"
#include "ModelPool.h"
#include "StatsMatrix.h"


struct AbcResult {
    IndelModelConfig winningConfig;
    Eigen::VectorXd meanParams; // 5 values: root, insRate, delRate, insLenParam, delLenParam
};

// Extract an arbitrary row/col subset into a fresh matrix.
inline Eigen::MatrixXd extractSubMatrix(const Eigen::MatrixXd& src,
                                         const std::vector<int>& rows,
                                         const std::vector<int>& cols) {
    Eigen::MatrixXd out(rows.size(), cols.size());
    for (size_t r = 0; r < rows.size(); ++r)
        for (size_t c = 0; c < cols.size(); ++c)
            out(r, c) = src(rows[r], cols[c]);
    return out;
}

// Stat-column groupings (0-indexed within the 26 stats-only columns).
// Root length: sequence-length stats. Rates: length stats + gap-count/position stats.
// Length-dist params: gap-size-shape stats.
struct ParamGroup {
    std::vector<int> statCols;
    std::vector<int> paramCols;
};

inline const std::vector<ParamGroup>& paramGroups() {
    static const std::vector<ParamGroup> groups = {
        { {0,1,2},                                                    {0} },       // root length
        { {0,1,2,3,9,22,23,24,25},                                    {1,2} },     // ins/del rate
        { {4,5,6,7,8,9,12,15,18,21,22,25},                            {3} },       // ins length param (POS_N-1 evidence)
        { {4,5,6,7,8,9,10,11,13,14,16,17,19,20,22,23,24},             {4} }        // del length param (POS_1 + POS_2 evidence)
    };
    return groups;
}

inline AbcResult abcInfer(const ModelPool& pool, const Eigen::VectorXd& empiricalStats, int topCutoff) {
    StatsMatrix sm{pool.pooledMatrix.rightCols(26)};
    Eigen::VectorXd dists = sm.mahalanobis(empiricalStats);

    // 1. vote using mixed top_cutoff across all models (full stat set, unchanged)
    std::vector<int> voteIdx = sm.nsmallest(dists, topCutoff);

    std::vector<int> voteCounts(pool.configs.size(), 0);
    for (int idx : voteIdx) {
        voteCounts[pool.tags[idx]]++;
    }

    int winningTag = 0;
    for (size_t k = 1; k < pool.configs.size(); ++k) {
        if (voteCounts[k] > voteCounts[winningTag] ||
            (voteCounts[k] == voteCounts[winningTag] &&
             pool.configs[k].complexity() < pool.configs[winningTag].complexity())) {
            winningTag = static_cast<int>(k);
        }
    }

    // 2. winning model's own rows (full pool, not just voteIdx)
    std::vector<int> modelRows;
    for (size_t i = 0; i < pool.tags.size(); ++i) {
        if (pool.tags[i] == winningTag) {
            modelRows.push_back(static_cast<int>(i));
        }
    }

    // 3. per-parameter-group: own stat subset, own Mahalanobis, own nearest-topCutoff, own mean
    Eigen::VectorXd meanParams = Eigen::VectorXd::Zero(NUM_PARAM_COLS);
    Eigen::MatrixXd statsOnly = pool.pooledMatrix.rightCols(26);

    for (const auto& group : paramGroups()) {
        Eigen::MatrixXd subMatrix = extractSubMatrix(statsOnly, modelRows, group.statCols);
        StatsMatrix smGroup{subMatrix};

        Eigen::VectorXd query(group.statCols.size());
        for (size_t c = 0; c < group.statCols.size(); ++c) query(c) = empiricalStats(group.statCols[c]);

        Eigen::VectorXd groupDists = smGroup.mahalanobis(query);
        int finalCutoff = std::min(topCutoff, static_cast<int>(modelRows.size()));
        std::vector<int> nearest = smGroup.nsmallest(groupDists, finalCutoff);

        for (int paramCol : group.paramCols) {
            double sum = 0.0;
            for (int localIdx : nearest) sum += pool.pooledMatrix(modelRows[localIdx], paramCol);
            meanParams(paramCol) = sum / double(nearest.size());
        }
    }

    return AbcResult{pool.configs[winningTag], meanParams};
}