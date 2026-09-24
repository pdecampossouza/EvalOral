# Data provenance and scope

The image material bundled here is a cleaned research snapshot derived from the public SBBrasil 2023 calibration/training materials previously organized by the upstream `Pipeline-for-Oral-Health-Images` repository.

## Canonical records used by EvalOral

The reproducibility package intentionally separates **source-module records** from generated copies used by the older repository for visualization or manual organization.

- 441 source-module image records.
- 257 SHA-256 content-unique photographs for label-free clustering.
- 121 DAI photographs with explicitly released view labels: 56 frontal and 65 occlusal.
- Six source contexts: permanent-dentition CPOD, permanent-dentition DAI, deciduous-dentition CEOD, deciduous occlusion, PUFA training, and trauma training.

The source-module directory name is treated as provenance/context, not automatically as a clinical diagnosis for every image.

## Why content deduplication matters

Some photographs are intentionally reused across calibration modules. Label-free clustering is therefore run on one representative per SHA-256 content hash so that identical image content is not overweighted.

## Licensing note

The MIT license in this repository applies to the software written for the project. The image materials originate from the upstream/public calibration resources and remain subject to their source terms and applicable public-data conditions. Redistribution of the images here is for reproducibility of the reported research; downstream users should verify institutional and source-specific requirements before republishing the image corpus elsewhere.
