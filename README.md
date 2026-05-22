# Stable Single-Pixel Contrastive Learning for Semantic and Geometric Tasks

This repository contains the official implementation accompanying the paper:

> **Stable Single-Pixel Contrastive Learning for Semantic and Geometric Tasks**  
> Leonid Pogorelyuk, Niels Bracher, Aaron Verkleeren, Lars Kühmichel, Stefan T. Radev  
> NeurIPS 2025 Workshop on Unifying Representations (UniReps)

## Overview

Many computer vision methods learn representations that are either:

- **semantic**, capturing object identity and meaning, or
- **geometric**, capturing precise spatial correspondences.

This work explores a family of **stable pixel-level contrastive losses** that jointly encode both types of information within a single representation. The proposed approach maps every pixel to an **overcomplete descriptor** that is simultaneously semantic and invariant to viewpoint changes.

## Citation

```bibtex
@article{pogorelyuk2025stable,
  title={Stable Single-Pixel Contrastive Learning for Semantic and Geometric Tasks},
  author={Pogorelyuk, Leonid and Bracher, Niels and Verkleeren, Aaron and Kühmichel, Lars and Radev, Stefan T.},
  journal={NeurIPS Workshop on Unifying Representations (UniReps)},
  year={2025},
  eprint={2512.04970},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
