#pragma once
#include <Eigen/Dense>
#include "../libs/Phylolib/includes/tree.h"
#include "../libs/Sailfish-backend/src/IndelSimulator.h"
#include "../libs/Sailfish-backend/src/MSA.h"
#include "../libs/Sailfish-backend/libs/sfc/sfc64.h"

#include "../libs/MSAStats/src/MsaStatsCalculator.h"


#include "Configs.h"
#include "LengthDistributions.h"


constexpr int NUM_PARAM_COLS = 5;

struct SimulationBatch {
    const IndelModelConfig& modelConfig;
    const PriorConfig& priorConfig;
    int numSims;
    unsigned seed;

    Eigen::MatrixXd run(tree& tree_) const {

        if (numSims <= 100) {
            throw std::invalid_argument("Number of simulations must be more than 10,000.");
        }
        Eigen::MatrixXd result(numSims, NUM_PARAM_COLS + 26);


        SimulationContext<SFC64> simContext(&tree_, seed);
        SimulationProtocol protocol(simContext.getTree()->getNodesNum() - 1);
        protocol.setMinSequenceSize(0);
        protocol.setSiteRateModel(SiteRateModel::SIMPLE);

        simContext.setProtocol(&protocol);

        auto& rng = simContext.getRng();
        const size_t numBranches = tree_.getNodesNum() - 1;
 
        // allocated once, sizes never change across iterations - reused in place
        std::vector<double> insertionRates(numBranches);
        std::vector<double> deletionRates(numBranches);
        std::vector<DiscreteDistribution*> insertionDists(numBranches);
        std::vector<DiscreteDistribution*> deletionDists(numBranches);
        std::vector<double> insertionProbs(priorConfig.lengthTruncation);
        std::vector<double> deletionProbs(priorConfig.lengthTruncation);
 
        for (int i = 0; i < numSims; ++i) {
 
            double sumOfRates = priorConfig.sampleSumRates(rng);
            double ratioOfRate = 1.0;
            if (!modelConfig.lockRates) {
                ratioOfRate = priorConfig.sampleRatioRates(rng);
            }
 
            double insertionRate = sumOfRates / (1.0 + ratioOfRate);
            double deletionRate = sumOfRates - insertionRate;
 
            fill(insertionRates.begin(), insertionRates.end(), insertionRate);
            fill(deletionRates.begin(), deletionRates.end(), deletionRate);
 
            double lengthParamInsertion = priorConfig.sampleLengthParam(rng);
            buildLengthDistProbs(priorConfig.lengthDist, lengthParamInsertion, priorConfig.lengthTruncation, insertionProbs);
            DiscreteDistribution Ins(insertionProbs);
 
            double lengthParamDeletion = lengthParamInsertion;
            if (!modelConfig.lockLengths) {
                lengthParamDeletion = priorConfig.sampleLengthParam(rng);
                buildLengthDistProbs(priorConfig.lengthDist, lengthParamDeletion, priorConfig.lengthTruncation, deletionProbs);
            } else {
                deletionProbs = insertionProbs;
            }
            DiscreteDistribution Del(deletionProbs);
 
            fill(insertionDists.begin(), insertionDists.end(), &Ins);
            fill(deletionDists.begin(), deletionDists.end(), &Del);
 
            protocol.setInsertionLengthDistributions(insertionDists);
            protocol.setDeletionLengthDistributions(deletionDists);
            protocol.setInsertionRates(insertionRates);
            protocol.setDeletionRates(deletionRates);
            protocol.setMaxInsertionLength(priorConfig.lengthTruncation);
 
            int rootLength = priorConfig.sampleRootLength(rng);
            protocol.setSequenceSize(rootLength);
 
            IndelSimulator<SFC64> indelSim(simContext, &protocol);
            auto eventMap = indelSim.generateSimulation();
 
            MSA<SFC64> msa = MSA<SFC64>(eventMap, simContext);
 
            MsaStatsCalculator msaStats(*msa.getSparseMSA(), msa.getNumberOfSequences(), msa.getMSAlength());
            msaStats.recomputeStats();
 
            auto stats = msaStats.getStatVec();
 
            result(i, 0) = rootLength;
            result(i, 1) = insertionRate;
            result(i, 2) = deletionRate;
            result(i, 3) = lengthParamInsertion;
            result(i, 4) = lengthParamDeletion;
 
            for (size_t j = 1; j < stats.size(); ++j) {
                result(i, NUM_PARAM_COLS + j - 1) = stats[j];
            }
        }
 
        return result;
    }
};

