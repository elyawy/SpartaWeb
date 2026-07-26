#pragma once

#include <string>
#include <utility>
#include <random>
#include <cmath>

struct IndelModelConfig {
    bool lockRates;
    bool lockLengths;
    std::string label;
    int complexity() const { return !lockRates + !lockLengths; }

};


struct PriorConfig {
    std::pair<int,int> rootLengthRange;
    std::pair<double,double> sumRatesRange;
    std::pair<double,double> ratioRatesRange; // (log-uniform bounds)
    std::pair<double,double> lengthParamRange;
    int lengthTruncation;

    enum class LengthDist { Zipf, Geometric, Poisson };
    LengthDist lengthDist = LengthDist::Zipf; // (default Zipf, others stub for later)

    template<typename RngType = std::mt19937_64>
    int sampleRootLength(RngType& rng) const {
        std::uniform_int_distribution<int> dist(rootLengthRange.first, rootLengthRange.second);
        return dist(rng);
    }

    template<typename RngType = std::mt19937_64>
    double sampleSumRates(RngType& rng) const {
        std::uniform_real_distribution<double> dist(sumRatesRange.first, sumRatesRange.second);
        return dist(rng);
    }

    template<typename RngType = std::mt19937_64>
    double sampleRatioRates(RngType& rng) const {
        // log-uniform distribution for ratio of insertion to deletion rates
        double logMin = std::log(ratioRatesRange.first);
        double logMax = std::log(ratioRatesRange.second);
        std::uniform_real_distribution<double> dist(logMin, logMax);
        return std::exp(dist(rng));
    }

    template<typename RngType = std::mt19937_64>
    double sampleLengthParam(RngType& rng) const {
        std::uniform_real_distribution<double> dist(lengthParamRange.first, lengthParamRange.second);
        return dist(rng);
    }


};