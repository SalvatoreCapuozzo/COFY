# Demo images: sources and licences

The images in this folder are small samples of the three use cases of the study, kept for the demo.
They are **not** covered by the code licence (AGPL-3.0, `LICENSE` at the repository root); each set
keeps the licence of its source, below.

Changes made to every image: resized so that the longer side is at most 1024 px and re-encoded as
JPEG. Labels: converted to YOLO format and split by node, each node keeping only its own classes
(`classes.yaml`), as in the study.

| Folder | Source | Licence |
|---|---|---|
| `wbc/` | Images: PBC dataset, A. Acevedo et al., *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems*, Data in Brief 30 (2020) 105474, https://doi.org/10.1016/j.dib.2020.105474. Boxes: https://github.com/medmabcf/White-Blood-Cell-Detection-Dataset | CC BY 4.0 |
| `dentex/` | DENTEX, I. E. Hamamci et al., *DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray*, https://doi.org/10.48550/arXiv.2305.19112 (enumeration subset) | CC BY-NC-SA 4.0: non-commercial use only; adaptations must be shared under the same licence |
| `kfm12/` | The authors' own images, acquired with the Kubic FLOTAC Microscope (G. Cringoli et al., Parasitology 148 (2021) 427–434, https://doi.org/10.1017/S003118202000219X). University of Naples Federico II | CC BY 4.0 |
