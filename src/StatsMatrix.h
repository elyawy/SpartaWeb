#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>


struct StatsMatrix {
    const Eigen::MatrixXd& statsOnly; // rows = sims, cols = stats only (no param cols)

    Eigen::VectorXd mahalanobis(const Eigen::VectorXd& query) const {
        const Eigen::RowVectorXd mean = statsOnly.colwise().mean();
        const Eigen::MatrixXd centered = statsOnly.rowwise() - mean;

        Eigen::MatrixXd cov = (centered.transpose() * centered) / double(statsOnly.rows() - 1);
        cov += Eigen::MatrixXd::Identity(cov.rows(), cov.cols()) * 1e-4;

        const Eigen::MatrixXd invCov = cov.inverse();

        Eigen::VectorXd dists(statsOnly.rows());
        for (int i = 0; i < statsOnly.rows(); ++i) {
            Eigen::RowVectorXd diff = statsOnly.row(i) - query.transpose();
            dists(i) = std::sqrt((diff * invCov * diff.transpose())(0, 0));
        }
        return dists;
    }

    std::vector<int> nsmallest(const Eigen::VectorXd& dists, int topCutoff) const {
        std::vector<int> indices(dists.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::partial_sort(indices.begin(), indices.begin() + topCutoff, indices.end(),
            [&dists](int a, int b) { return dists(a) < dists(b); });

        indices.resize(topCutoff);
        return indices;
    }
};