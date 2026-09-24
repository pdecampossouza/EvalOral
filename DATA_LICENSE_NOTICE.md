# Data and licensing notice

The MIT license in this repository applies to original project software and project-authored documentation unless a file states otherwise.

The oral photographs under `data/raw/` are a research snapshot derived from public SBBrasil 2023 calibration/training materials and from the upstream `Pipeline-for-Oral-Health-Images` organization of those public materials. They are **not relicensed under MIT by this repository**. Their reuse and redistribution remain subject to the terms, provenance, and applicable public-data conditions of the source materials.

Before publishing a public fork that includes `data/raw/`, downstream users should verify that their intended redistribution is permitted. If image redistribution is not desired, the code can be published without `data/raw/`; the manifests and feature arrays remain useful for many software and methodological checks, although image-dependent galleries and Grad-CAM regeneration will then require the images to be restored locally.

See `docs/DATA_PROVENANCE.md` for corpus construction and deduplication details.
