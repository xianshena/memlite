# Recorded results, controls and limits

## Main 308M numerical household experiment

Each seed: 308,487,175 trainable parameters, 12 epochs, 948 updates, 10,000 independent train contexts presented 120,000 times. Test scenes are shared across seeds. Daily counts: 1506/1600, 1505/1600, 1506/1600; unweighted seed mean 94.10417%. Full tie-aware test scores are 95.30%, 95.25%, 95.30%, because the 2,000 tests also contain 300 history examples and 100 unknown utility ties. Do not use full tie-aware accuracy as hidden-preference accuracy.

History control: 200 PAIRS, each with an original and opposite history. Both must be correct and choose different actions: 200/200 pairs for every seed. Current non-memory columns and feedback sentinels are identical. Clearing memory scores only the original-side 200 examples: 102,100,100 correct. This differs from the 300 history rows in the frozen test set. Full pair-level predictions were not retained for all 200 pairs in the selected records; aggregate diagnostics and 12 action examples are supplied, and the diagnostic implementation is included. Do not invent missing pair-level raw outputs.

Separate 2,000 unknown-preference controls: 994,958,1002 correct; binary NLL .694304/.697036/.697372; mean abs(p-.5) .017036/.026782/.024390; max probability >=.9 fraction zero. Uniform reference: NLL ln2 and binary Brier .25. Raw ECE on the 1,900 known test scenes, 15 equal-frequency bins: .022685/.025789/.026018; no temperature fit.

BF16 candidate-reversal maximum probability differences: 1.1772243e-4,6.4743450e-5,2.9785093e-5. All miss the earlier object-task G1b 1e-6 reference. The household protocol registered a permutation control but no independent numerical pass threshold. Tiny GPU fixtures used a different atol; passing those is not full-model strict invariance.

All 2,000 selected actions per seed arrived/completed in the deterministic simulator; BFS supplies paths and abstract actions. Positive teacher-regret failures: 94,95,94. `failure_examples.json` preserves selected errors (up to five high-regret examples per seed), including choosing entertainment rather than drinking in a thirsty scene. Success at completing a suboptimal activity does not make the choice correct.

Historical CUDA allocated peak: 6,845,809,152 B = 6.845809152 GB = 6.375657 GiB; reserved: 7,245,660,160 B = 7.24566016 GB = 6.748047 GiB. These are allocator counters, not total shared system RAM. Batch64 forward timing is not single-example latency. See JSON evidence rather than extrapolating hardware throughput.

## CPU object-memory G0/G1a

22 historical tests passed. Explicit experience/feedback-swap/permutation/subject-isolation controls: 100/100 pilot episodes; no-experience forced choice 983/2000. No-memory/last-observation/clear/shuffled pilot controls: 48/100. Ideal known coordinate error P95=0 is not physical localization accuracy. The state table has zero trainable parameters and diagnostic identities, not learned vision.

Capacity diagnostics use only ten seeds per cell. At 256 distractors all budgets (32/64/192 KiB CPython heap state) retained only 4/10 forced choices; at 32 KiB and 64 distractors also 4/10. Oldest-write eviction loses feedback. This is not formal G3 acceptance. G1b and G2–G6 remain unverified. CPU QA during release preparation is separately labeled and does not replace historical numbers.

## Exploratory appendix: Qwen teacher / 1.85B student

Teacher inference annotates already-existing numerical scenes, returning a legal index and short reason. The program derives a one-hot label; it is not calibrated teacher confidence or human truth. Final student stage: 1,850,631,175 parameters, depth144, 272 training examples loaded, 64 validation, 3 epochs/9 steps, 42/64 (65.625%) agreement with the teacher. 303 accepted train annotations were cumulative, not all consumed by this final stage. No independent test, new-model history/clear/permutation evaluation or comparative size study was completed.

First expansion attempt: five scales each six optimizer steps; the first 1.85B checkpoint failed for insufficient disk. The recovery stage inherited a saved depth112 prefix, added random blocks, reset optimizer, and trained nine steps. Total executed updates 39 include six from the unsaved stage; do not claim those six are preserved in the final model. Five saved artifacts were BF16 weights only, not optimizer/RNG exact-resume states. Final CUDA allocated/reserved peaks: 40,308,602,880 / 41,999,663,104 bytes; approximately 42 decimal GB is reserved memory, not 42 GB useful tensors.

This appendix does not inherit the main 308M results, prove that larger models help, or establish real perception, biology, autonomous memory, Jev/RLCD reproduction or community adoption. Raw teacher replies and operations are deliberately excluded from the source distribution pending separate rights/scope review.
