#include <emscripten/bind.h>
#include <chrono>
#include <string>
#include <vector>

#include "Configs.h"
#include "LengthDistributions.h"
#include "ModelPool.h"
#include "AbcInference.h"

#include "../libs/MSAStats/src/MsaStatsCalculator.h"

using namespace emscripten;

struct InferenceResult {
    std::string model;
    std::vector<double> params; // root, insRate, delRate, insLenParam, delLenParam
};

// Hardcoded prior defaults (matches spartaabc/default_prior.json / test files)
static PriorConfig makeDefaultPrior(int rootMin, int rootMax) {
    PriorConfig prior;
    prior.rootLengthRange = {rootMin, rootMax};
    prior.sumRatesRange = {0.0, 0.05};
    prior.ratioRatesRange = {0.1, 10.0};
    prior.lengthParamRange = {1.01, 2.0};
    prior.lengthTruncation = 50;
    prior.lengthDist = PriorConfig::LengthDist::Zipf;
    return prior;
}

// Runs a small SIM-only batch and returns ms-per-sim, used by JS to estimate
// total runtime before kicking off the real (slow) run.
double benchmarkSimTime(std::string newickText, int sampleSims, unsigned seed) {
    tree tree_(newickText, false);

    IndelModelConfig simConfig{true, true, "SIM"};
    PriorConfig prior = makeDefaultPrior(100, 500); // placeholder range, cost only depends weakly on this

    SimulationBatch batch{simConfig, prior, sampleSims, seed};

    auto start = std::chrono::high_resolution_clock::now();
    batch.run(tree_);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsedMs / sampleSims;
}

InferenceResult runInference(std::vector<std::string> msaSeqs,
                              std::string newickText,
                              int numSimsPerModel,
                              unsigned seed) {
    // empirical stats from user MSA
    MsaStatsCalculator msaStats(msaSeqs);
    msaStats.recomputeStats();
    auto fullStats = msaStats.getStatVec(); // 27 values, index0 = avg gap

    Eigen::VectorXd empiricalStats(26);
    for (size_t j = 1; j < fullStats.size(); ++j) empiricalStats(j - 1) = fullStats[j];

    double minLen = fullStats[3]; // MSA_MIN_LEN
    double maxLen = fullStats[2]; // MSA_MAX_LEN

    PriorConfig prior = makeDefaultPrior(int(minLen * 0.8), int(maxLen * 1.1));

    tree tree_(newickText, false);

    IndelModelConfig simConfig{true, true, "SIM"};
    IndelModelConfig rimConfig{false, false, "RIM"};

    ModelPool pool{{simConfig, rimConfig}, prior, numSimsPerModel, seed};
    pool.run(tree_);

    AbcResult res = abcInfer(pool, empiricalStats, /*topCutoff=*/100);

    InferenceResult out;
    out.model = res.winningConfig.label;
    out.params.resize(NUM_PARAM_COLS);
    for (int i = 0; i < NUM_PARAM_COLS; ++i) out.params[i] = res.meanParams(i);

    return out;
}

EMSCRIPTEN_BINDINGS(sparta_module) {
    register_vector<std::string>("StringVector");
    register_vector<double>("DoubleVector");

    value_object<InferenceResult>("InferenceResult")
        .field("model", &InferenceResult::model)
        .field("params", &InferenceResult::params);

    emscripten::function("benchmarkSimTime", &benchmarkSimTime);
    emscripten::function("runInference", &runInference);

    class_<tree>("Tree")
        // Constructor accepts newick string OR file path — same as C++ side
        .constructor<const std::string&, bool>()
        .property("num_nodes", &tree::getNodesNum)
        .property("root",      &tree::getRoot, allow_raw_pointers());
}