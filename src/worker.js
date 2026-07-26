import createSparta from './dist/sparta.js';

let Module = null;

self.onmessage = async (e) => {
    if (!Module) {
        Module = await createSparta();
    }

    const { msaSeqs, treeText, numSimsPerModel, seed } = e.data;

    // quick benchmark to estimate total time (SIM config, small sample)
    const msPerSim = Module.benchmarkSimTime(treeText, 150, seed);
    const estimatedMs = msPerSim * numSimsPerModel * 2; // SIM + RIM
    self.postMessage({ type: "eta", estimatedMs });

    // real run (slow, blocks this worker thread only)
    const seqVec = new Module.StringVector();
    msaSeqs.forEach(s => seqVec.push_back(s));

    const result = Module.runInference(seqVec, treeText, numSimsPerModel, seed);
    seqVec.delete();

    const params = [];
    for (let i = 0; i < result.params.size(); i++) {
        params.push(result.params.get(i));
    }
    result.params.delete();

    self.postMessage({ type: "done", model: result.model, params });
};
