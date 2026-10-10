# NOTE on m2_result.txt / m2_perclass.txt

These two files were produced in an earlier development round (August) and use an
older checkpoint set. Direct re-evaluation of the checkpoints released here gives
different numbers for the single-operator variants, e.g. sat-only mAP50 is 0.8116
(m2_result.txt: 0.809) and wb-only is 0.8092 at seed 0 (not reported there).

For any comparison, use the files in results_per_seed/ instead:

- results_per_seed/results_dump/*_seed{0,1,2}.json : per-seed, per-class AP50 for all
  reported configurations (raw, UEP, UEP+LCA, wb-only, sat-only, CLAHE-only)
- results_per_seed/per_seed_summary.txt : the 3-seed means/sds and the cross-evaluation
  (raw-trained model on enhanced test set: 0.5103 +/- 0.0189; UEP-trained on raw test:
  0.6414 +/- 0.0510)
- results_per_seed/p7_3_summary.txt : 3-seed means/sds for the six ablation configurations
- results_per_seed/latency_bench.json : hardware, library versions and the timing protocol
  (UEP 27.8 ms single-thread CPU, median of 200 runs; detection 6.05 ms on RTX 4090,
  median of 50 runs)

The m2_* files are kept for historical traceability only.
