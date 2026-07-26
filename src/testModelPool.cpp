#include <iostream>
#include <algorithm>

#include "../libs/Phylolib/includes/tree.h"
#include "Configs.h"
#include "ModelPool.h"

int main() {
    tree tree_("../libs/Sailfish-backend/tests/trees/normalbranches_nLeaves100.treefile");

    IndelModelConfig simConfig;
    simConfig.lockRates = true;
    simConfig.lockLengths = true;
    simConfig.label = "SIM";

    IndelModelConfig rimConfig;
    rimConfig.lockRates = false;
    rimConfig.lockLengths = false;
    rimConfig.label = "RIM";

    PriorConfig prior;
    prior.rootLengthRange = {100, 500};
    prior.sumRatesRange = {0.0, 0.05};
    prior.ratioRatesRange = {0.1, 10.0};
    prior.lengthParamRange = {1.01, 2.0};
    prior.lengthTruncation = 50;
    prior.lengthDist = PriorConfig::LengthDist::Zipf;

    ModelPool pool{{simConfig, rimConfig}, prior, /*numSimsPerModel=*/10001, /*seed=*/123456789};

    std::cout << "Running pool...\n";
    pool.run(tree_);

    std::cout << "Pooled matrix: " << pool.pooledMatrix.rows() << " x " << pool.pooledMatrix.cols() << "\n";
    std::cout << "Tags size: " << pool.tags.size() << "\n";

    int countSim = std::count(pool.tags.begin(), pool.tags.end(), 0);
    int countRim = std::count(pool.tags.begin(), pool.tags.end(), 1);
    std::cout << "Tag 0 (SIM) count: " << countSim << "\n";
    std::cout << "Tag 1 (RIM) count: " << countRim << "\n";

    // sanity: SIM rows should have insertionRate == deletionRate (col 1 == col 2)
    // RIM rows likely differ
    std::cout << "SIM row 0 rates: " << pool.pooledMatrix(0, 1) << " " << pool.pooledMatrix(0, 2) << "\n";
    std::cout << "RIM row 0 rates: " << pool.pooledMatrix(10001, 1) << " " << pool.pooledMatrix(10001, 2) << "\n";

    return 0;
}