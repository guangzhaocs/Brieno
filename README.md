<h2 align="center">
  <img src="media/brieno_logo_animated.svg" alt="Brieno" width="560"><br>
  A foundation model for raw nanopore signal through discrete tokenization
</h2>

[![bioRxiv](https://img.shields.io/badge/bioRxiv-TBA-green)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/guangzhaocs/Brieno/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/brieno)](https://pypi.org/project/brieno/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Brieno-yellow)](https://huggingface.co/guangzhaocs/Brieno)
<!-- Enable once the repository is public:
[![Release](https://img.shields.io/github/v/release/guangzhaocs/Brieno?include_prereleases)](https://github.com/guangzhaocs/Brieno/releases)
[![Downloads](https://img.shields.io/pepy/dt/brieno?color=97ca00)]()
-->

## Overview

**Brieno** is a self-supervised model that learns reusable representations of raw
nanopore signals. It first converts raw signals into discrete tokens through residual
vector quantization and then pretrains a Transformer to predict masked tokens from their
context. The pretrained Transformer can be fine-tuned with task-specific heads for
multiple downstream tasks.

<p align="center">
  <img src="media/model_overview.png" alt="Brieno pipeline: tokenizer pretraining with residual vector quantization, masked-token pretraining of the transformer, and fine-tuning on downstream tasks" width="100%">
</p>

With frozen or fine-tuned features, the model:

- identifies **species** without requiring reference genomes;
- outperforms Dorado in **native-barcode demultiplexing**;
- matches Dorado's **basecalling** performance on bacteriophage lambda;
- approaches Dorado's specialized model on **DNA methylation** calling.

## Name origin

*Brieno* is a shortened form of [*Brienomyrus*](https://en.wikipedia.org/wiki/Brienomyrus), a genus of weakly
electric fish that communicate through patterns of electric pulses. The name echoes Oxford
Nanopore's fish-inspired tool names, such as Bonito and Dorado, and also reflects the
model's aim to learn informative patterns in the ionic current signals recorded during
nanopore sequencing.

## Citation

Citation details will be added once the manuscript is available. Until then, please
refer to this repository.

## License

Released under the [Apache License 2.0](LICENSE).
