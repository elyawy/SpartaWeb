#include <iostream>

#include "../libs/Phylolib/includes/tree.h"
#include "Configs.h"
#include "SimulationBatch.h"
#include "StatsMatrix.h"

int main() {
    tree tree_("../libs/Sailfish-backend/tests/trees/normalbranches_nLeaves100.treefile");

    IndelModelConfig simConfig;
    simConfig.lockRates = true;
    simConfig.lockLengths = true;
    simConfig.label = "SIM";

    PriorConfig prior;
    prior.rootLengthRange = {100, 500};
    prior.sumRatesRange = {0.0, 0.05};
    prior.ratioRatesRange = {0.1, 10.0};
    prior.lengthParamRange = {1.01, 2.0};
    prior.lengthTruncation = 50;
    prior.lengthDist = PriorConfig::LengthDist::Zipf;

    SimulationBatch batch{simConfig, prior, /*numSims=*/10001, /*seed=*/123456789};

    std::cout << "Running batch...\n";
    Eigen::MatrixXd full = batch.run(tree_);
    std::cout << "Full matrix: " << full.rows() << " x " << full.cols() << "\n";

    Eigen::MatrixXd statsOnly = full.rightCols(26);
    StatsMatrix sm{statsOnly};

    // sanity check: query = row 42's own stats, should be closest to itself
    int testRow = 42;
    Eigen::VectorXd query = statsOnly.row(testRow).transpose();

    Eigen::VectorXd dists = sm.mahalanobis(query);
    std::cout << "Distance to self (row " << testRow << "): " << dists(testRow) << "\n";

    std::vector<int> top5 = sm.nsmallest(dists, 5);
    std::cout << "Top 5 nearest indices: ";
    for (int idx : top5) std::cout << idx << " ";
    std::cout << "\n";

    bool selfFound = std::find(top5.begin(), top5.end(), testRow) != top5.end();
    std::cout << "Self found in top5? " << selfFound << "\n";

    return 0;
}