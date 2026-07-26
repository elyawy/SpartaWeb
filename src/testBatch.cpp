#include <iostream>

#include "../libs/Phylolib/includes/tree.h"
#include "Configs.h"
#include "SimulationBatch.h"

int main() {
    tree tree_("../libs/Sailfish-backend/tests/trees/normalbranches_nLeaves100.treefile");

    IndelModelConfig simConfig;
    simConfig.lockRates = true;
    simConfig.lockLengths = true;
    simConfig.label = "SIM";

    PriorConfig prior;
    prior.rootLengthRange = {100, 500};
    prior.sumRatesRange = {0.0, 0.05};
    prior.ratioRatesRange = {0.1, 10.0}; // linear-space bounds, sampler logs internally
    prior.lengthParamRange = {1.01, 2.0};
    prior.lengthTruncation = 50;
    prior.lengthDist = PriorConfig::LengthDist::Zipf;

    SimulationBatch batch{simConfig, prior, /*numSims=*/10001, /*seed=*/12345678};

    std::cout << "Running batch...\n";
    Eigen::MatrixXd result = batch.run(tree_);

    std::cout << "Matrix shape: " << result.rows() << " x " << result.cols() << "\n";
    std::cout << "First row: " << result.row(0) << "\n";
    std::cout << "Any NaN? " << result.array().isNaN().any() << "\n";

    return 0;
}