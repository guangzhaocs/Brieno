<h2 align="center">
  <img src="media/brieno_logo_animated.svg" alt="Brieno" width="560"><br>
  A foundation model for raw nanopore signal through discrete tokenization
</h2>

[![bioRxiv](https://img.shields.io/badge/bioRxiv-TBA-green)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/guangzhaocs/Brieno/blob/main/LICENSE)
<!-- Enable once the repository is public and the package is on PyPI:
[![Release](https://img.shields.io/github/v/release/guangzhaocs/Brieno?include_prereleases)](https://github.com/guangzhaocs/Brieno/releases)
[![PyPI](https://img.shields.io/pypi/v/brieno)]()
[![Downloads](https://img.shields.io/pepy/dt/brieno?color=97ca00)]()
-->

## Overview

Nanopore sequencing reads nucleic acids as changes in ionic current that carry
information beyond the base sequence. Self-supervised pretrained models have advanced
genomic sequence modeling and single-cell analysis, but remain underexplored for raw
nanopore signals, which are continuous and lack a native token vocabulary.

**Brieno** combines reconstruction-based tokenization through residual vector
quantization with a transformer for contextual representation learning. The transformer
is pretrained with masked-token prediction on signals from **118 million reads**.

<p align="center">
  <img src="media/model_overview.png" alt="Brieno pipeline: tokenizer pretraining with residual vector quantization, masked-token pretraining of the transformer, and fine-tuning on downstream tasks" width="100%">
</p>

With frozen or fine-tuned features, the model:

- identifies **species** without requiring reference genomes;
- outperforms Dorado in **native-barcode demultiplexing**;
- matches Dorado's **basecalling** performance on bacteriophage lambda;
- approaches Dorado's specialized model on **DNA methylation** calling.

Pretraining also improves the use of contextual information and generalization across
sequencing runs. Together, these findings support self-supervised pretraining as a basis
for transferable representations of raw nanopore signals.

## Name origin

*Brieno* is a shortened form of [*Brienomyrus*](https://en.wikipedia.org/wiki/Brienomyrus), a genus of weakly electric fishes that
communicate through patterns of electric pulses — reading meaning from an electrical
signal rather than from a written sequence.

## Status

This repository is the public home of Brieno. Code, pretrained weights and usage
instructions are being prepared for release and will be added here. The manuscript
describing the method is currently under review.

## Topics

Nanopore sequencing · self-supervised learning · foundation models · raw signal
representation learning · vector quantization · species identification · metagenomics ·
DNA methylation

## Citation

Citation details will be added once the manuscript is available. Until then, please
refer to this repository.

## License

Released under the [Apache License 2.0](LICENSE).
