# MLML alignment — research direction, not endorsement

MLML publicly frames its goal around human-like general artificial intelligence and has recent work on compositional world models, model-based generalist agents, systematic generalization and explicit learned theories of the world.

Relevant research ideas for SAM-WM:

- **Dreamweaver**: compositional world models from pixels and explicit OOD datasets. Lesson: evaluate recombination/generalization rather than only in-distribution prediction.
- **Imagine the Unseen World**: systematic generalization in visual world models. Lesson: unseen combinations need dedicated evaluation.
- **Dr. Strategy**: model-based generalist agents and strategic dreaming. Lesson: imagined futures matter only if they improve action selection.
- **Facing Off World Model Backbones**: memory/efficiency comparisons across RNN/Transformer/S4. Lesson: architecture should be selected empirically, including compute efficiency.
- **Learning to Theorize the World from Observation / NEO**: explicit executable, compositional theories for explanation-driven generalization. Lesson: prediction alone is not the same as understanding.

SAM-WM is deliberately orthogonal rather than a copy. Its research question is: **how should a real-world action-conditioned urban world model represent counterfactual cooling outcomes when historical intervention support is uneven and climate/temperature regimes shift?**

What SAM-WM still needs before it can credibly stand beside this style of research:

1. a sharply stated research question;
2. real datasets with immutable provenance;
3. strong cheap baselines;
4. systematic/OOD splits;
5. ablations tied to claims;
6. compute/latency analysis;
7. uncertainty calibration;
8. real action-effect validation;
9. reproducible commands and released artifacts;
10. no SOTA claim unless a fair benchmark demonstrates it.

SAM-WM should not claim that MLML work "missed" urban cooling. Their papers solve different research problems. The defensible contribution is to transfer the lab's themes of compositional world understanding, generalization and planning into a real, support-aware physical intervention domain with causal uncertainty and deployment constraints.
