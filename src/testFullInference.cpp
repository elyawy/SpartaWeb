#include <iostream>
#include <vector>

#include "Configs.h"
#include "LengthDistributions.h"
#include "ModelPool.h"
#include "AbcInference.h"


int main() {
    tree tree_("../libs/Sailfish-backend/tests/trees/normalbranches_nLeaves100.treefile");

    // ---- 1. simulate ground truth MSA with known fixed params ----
    const double TRUE_INS_RATE = 0.007;
    const double TRUE_DEL_RATE = 0.035;
    const double TRUE_INS_LEN_PARAM = 1.53;
    const double TRUE_DEL_LEN_PARAM = 1.11;
    const int TRUE_ROOT_LENGTH = 500;
    const int TRUNCATION = 50;

    SimulationContext<SFC64> simContext(&tree_, /*seed=*/999);
    SimulationProtocol protocol(simContext.getTree()->getNodesNum() - 1);
    protocol.setMinSequenceSize(0);
    protocol.setSiteRateModel(SiteRateModel::SIMPLE);
    simContext.setProtocol(&protocol);

    size_t numBranches = tree_.getNodesNum() - 1;
    std::vector<double> insertionRates(numBranches, TRUE_INS_RATE);
    std::vector<double> deletionRates(numBranches, TRUE_DEL_RATE);

    std::vector<double> insertionProbs = buildLengthDistProbs(PriorConfig::LengthDist::Zipf, TRUE_INS_LEN_PARAM, TRUNCATION);
    std::vector<double> deletionProbs = buildLengthDistProbs(PriorConfig::LengthDist::Zipf, TRUE_DEL_LEN_PARAM, TRUNCATION);
    DiscreteDistribution Ins(insertionProbs);
    DiscreteDistribution Del(deletionProbs);
    std::vector<DiscreteDistribution*> insertionDists(numBranches, &Ins);
    std::vector<DiscreteDistribution*> deletionDists(numBranches, &Del);

    protocol.setInsertionLengthDistributions(insertionDists);
    protocol.setDeletionLengthDistributions(deletionDists);
    protocol.setInsertionRates(insertionRates);
    protocol.setDeletionRates(deletionRates);
    protocol.setMaxInsertionLength(TRUNCATION);
    protocol.setSequenceSize(TRUE_ROOT_LENGTH);

    IndelSimulator<SFC64> indelSim(simContext, &protocol);
    auto eventMap = indelSim.generateSimulation();
    MSA<SFC64> msa(eventMap, simContext);

    MsaStatsCalculator msaStats(*msa.getSparseMSA(), msa.getNumberOfSequences(), msa.getMSAlength());
    msaStats.recomputeStats();
    auto fullStats = msaStats.getStatVec(); // 27 values, index0 = avg gap

    // ---- 2. build empirical query vector (26 vals, drop avg gap) ----
    Eigen::VectorXd empiricalStats(26);
    for (size_t j = 1; j < fullStats.size(); ++j) empiricalStats(j - 1) = fullStats[j];

    double minLen = fullStats[3]; // MSA_MIN_LEN
    double maxLen = fullStats[2]; // MSA_MAX_LEN
    std::cout << "True MSA min/max seq len: " << minLen << " / " << maxLen << "\n";

    // ---- 3. derive prior root length range, like python's scale_factor [0.8,1.1] ----
    PriorConfig prior;
    prior.rootLengthRange = {int(minLen * 0.8), int(maxLen * 1.1)};
    prior.sumRatesRange = {0.0, 0.15};
    prior.ratioRatesRange = {0.1, 10.0};
    prior.lengthParamRange = {1.01, 2.0};
    prior.lengthTruncation = TRUNCATION;
    prior.lengthDist = PriorConfig::LengthDist::Zipf;

    std::cout << "Derived root length range: [" << prior.rootLengthRange.first
               << ", " << prior.rootLengthRange.second << "]\n";

    // ---- 4. run ModelPool (SIM + RIM) and infer ----
    IndelModelConfig simConfig{true, true, "SIM"};
    IndelModelConfig rimConfig{false, false, "RIM"};

    ModelPool pool{{simConfig, rimConfig}, prior, /*numSimsPerModel=*/15000, /*seed=*/42};
    std::cout << "Running pool...\n";
    pool.run(tree_);

    AbcResult res = abcInfer(pool, empiricalStats, /*topCutoff=*/100);

    std::cout << "\n--- Inference result ---\n";
    std::cout << "Winning model: " << res.winningConfig.label << "\n";
    std::cout << "root_length:   " << res.meanParams(0) << " (true " << TRUE_ROOT_LENGTH << ")\n";
    std::cout << "ins_rate:      " << res.meanParams(1) << " (true " << TRUE_INS_RATE << ")\n";
    std::cout << "del_rate:      " << res.meanParams(2) << " (true " << TRUE_DEL_RATE << ")\n";
    std::cout << "ins_len_param: " << res.meanParams(3) << " (true " << TRUE_INS_LEN_PARAM << ")\n";
    std::cout << "del_len_param: " << res.meanParams(4) << " (true " << TRUE_DEL_LEN_PARAM << ")\n";

    return 0;
}
