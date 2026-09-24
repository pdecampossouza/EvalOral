# How to publish this release on GitHub

The directory is already laid out as a repository root.

## 1. Review the two publication gates

Before pushing publicly:

1. resolve or explicitly amend the LinearSVC 0.964 manuscript-vs-rerun issue in `docs/REPRODUCIBILITY_NOTES.md`;
2. confirm that public redistribution of `data/raw/` is permitted for the intended repository. See `DATA_LICENSE_NOTICE.md`.

## 2. Create the repository

Create an empty GitHub repository, for example `EvalOral`, without auto-generating a README or license.

## 3. Initialize locally

```bash
git init
git add .
git commit -m "Initial EvalOral reproducibility release"
git branch -M main
git remote add origin https://github.com/<USER>/<REPOSITORY>.git
git push -u origin main
```

No file in the current full package exceeds GitHub's 100 MB single-file limit. The manuscript PDF is the largest file at approximately 43 MB.

## 4. Optional: publish code without raw photographs

If redistribution of the source images is not yet cleared, remove `data/raw/` before pushing and explain in the README where users can obtain/reconstruct the public source materials. Image-dependent experiments will then require users to restore those files locally.

## 5. Create a release

After the manuscript DOI is final, update `CITATION.cff` and the paper metadata, regenerate `SHA256SUMS.txt`, tag the release, and push the tag:

```bash
python scripts/make_release_manifest.py
git add CITATION.cff SHA256SUMS.txt
git commit -m "Finalize citation metadata"
git tag -a v1.0.0 -m "EvalOral v1.0.0"
git push origin main --tags
```

## 6. Recommended repository settings

- Enable GitHub Actions.
- Keep branch protection on `main` once the release is stable.
- Add the manuscript DOI under **About** after publication.
- Add topics such as `neuro-fuzzy`, `evolving-systems`, `dental-imaging`, `explainable-ai`, `grad-cam`, and `human-in-the-loop`.
