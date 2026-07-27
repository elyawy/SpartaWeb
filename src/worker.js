import createSparta from './dist/sparta.js';

let Module = null;

function validateInputs(msaSeqs, treeText, numSimsPerModel, priorOverrides) {
    if (!Array.isArray(msaSeqs) || msaSeqs.length < 2) {
        return "need at least 2 sequences in the MSA";
    }
    if (msaSeqs.some(s => !s || s.trim().length === 0)) {
        return "one or more sequences are empty";
    }
    const lengths = new Set(msaSeqs.map(s => s.length));
    if (lengths.size > 1) {
        return "sequences are not aligned (different lengths)";
    }
    if (!treeText || treeText.trim().length === 0) {
        return "tree (newick) text is empty";
    }
    if (!Number.isFinite(numSimsPerModel) || numSimsPerModel <= 0) {
        return "numSimsPerModel must be a positive number";
    }
    const [min, max] = priorOverrides?.sumRatesRange ?? [-1, -1];
    const isSentinel = min === -1 && max === -1;
    if (!isSentinel) {
        if (!Number.isFinite(min) || !Number.isFinite(max)) {
            return "sum of rates range must be numbers";
        }
        if (min < 0 || max > 2.0) {
            return "sum of rates range must be within [0, 2.0]";
        }
        if (min >= max) {
            return "sum of rates min must be less than max";
        }
    }
    return null;
}

self.onmessage = async (e) => {
    const { msaSeqs, treeText, numSimsPerModel, seed, priorOverrides } = e.data;

    const validationError = validateInputs(msaSeqs, treeText, numSimsPerModel, priorOverrides);
    if (validationError) {
        self.postMessage({ type: "error", message: validationError });
        return;
    }

    try {
        if (!Module) {
            Module = await createSparta();
        }

        // quick benchmark to estimate total time (SIM config, small sample)
        const msPerSim = Module.benchmarkSimTime(treeText, 150, seed);
        const estimatedMs = msPerSim * numSimsPerModel * 2; // SIM + RIM
        self.postMessage({ type: "eta", estimatedMs });

        // real run (slow, blocks this worker thread only)
        const seqVec = new Module.StringVector();
        msaSeqs.forEach(s => seqVec.push_back(s));

        const overrides = priorOverrides ?? { sumRatesRange: [-1, -1] };
        const result = Module.runInference(seqVec, treeText, numSimsPerModel, seed, overrides);
        seqVec.delete();

        const params = [];
        for (let i = 0; i < result.params.size(); i++) {
            params.push(result.params.get(i));
        }
        result.params.delete();

        self.postMessage({ type: "done", model: result.model, params });
    } catch (err) {
        self.postMessage({ type: "error", message: err?.message || String(err) });
    }
};