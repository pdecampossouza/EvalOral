# Model scope and non-claims

EvalOral is a research framework for **auditability of evolving rule-based image models**. It is not a clinical diagnostic device.

Important architectural boundaries:

1. The primary evolving neuro-fuzzy model does **not** receive CNN embeddings. Its released input is the 6,272-dimensional grayscale+edge descriptor, standardized and reduced to four whitened principal components.
2. The lightweight CNN is a **separate Grad-CAM audit branch**. The CNN and fuzzy model are linked post hoc through the identity of the same photograph.
3. Label-free K-means does **not** initialize or construct fuzzy rules in the primary model. It provides an independent visual-organization reference.
4. The anchor supervised task uses only the released frontal/occlusal labels. `lateral` is introduced only in the focused human-feedback experiments and is not retroactively treated as original ground truth.
5. PUFA, trauma, and source-module names are provenance/context in this paper, not complete image-level diagnostic labels.
6. The bundled anatomical HITL kits are a post-manuscript extension toward maxilla/mandible, side, and anterior/posterior semantics. They are deliberately separated from the reported manuscript results.
