# Traffic Sign Quantization Robustness

PyTorch code for training traffic-sign classifiers and evaluating inference under quantization and image corruption.

## Datasets

The project currently reads GTSRB classification data from `data/raw`. Other datasets below are real, publicly described sources that may support cross-dataset or scene-level evaluation. They are not downloaded or bundled here. Check each source's terms and citation requirements before use.

| Dataset | What it contains | Source | Fit for this project |
| --- | --- | --- | --- |
| German Traffic Sign Recognition Benchmark (GTSRB) | Cropped traffic-sign images across 43 classes | [Dataset page and download](https://benchmark.ini.rub.de/gtsrb_dataset.html) | Current classifier input format and primary benchmark. |
| BelgiumTSC | Cropped Belgian traffic signs across 62 classes | [ETH Zurich dataset page and downloads](https://btsd.ethz.ch/shareddata/) | Useful for testing geographic generalization; the current loader does not read its layout directly. |
| Mapillary Traffic Sign Dataset (MTSD) | Traffic signs in globally diverse street-level images, with detection and classification annotations | [Dataset access](https://www.mapillary.com/dataset/trafficsign) · [Dataset paper](https://arxiv.org/abs/1909.04422) | Useful for broader real-world conditions; its annotations and full-scene images need preprocessing before classifier evaluation. Access and usage follow Mapillary's terms. |
| BDD100K | Diverse driving images and videos with object annotations, including traffic signs | [Dataset portal](https://bdd-data.berkeley.edu/portal.html) · [Label format](https://github.com/bdd100k/bdd100k/blob/master/doc/format.md) | A possible scene-level source; its object labels are not directly compatible with the current 43-class classifier. |

The current `GTSRBDataset` expects the official GTSRB CSV and image layout. Download the data from its source and place it under `data/raw`; keep downloaded data out of Git.

## Local data checks

Set `GTSRB_DATA_ROOT` to the directory containing the official GTSRB files when running real-data integration tests. For the checkpoint-pair preflight, also set `FP32_CHECKPOINT` and `QUANTIZED_CHECKPOINT` to compatible weights produced for this project's model architecture.
