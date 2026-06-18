# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-18

### Added

- Core direction extraction via difference-in-means (DiM) with pluggable pairing strategies (`arditi`, `caa`)
- `SafetyCompassMonitor` for tracking concept drift during training
- `SafetyCompassCallback` for HuggingFace Trainer integration
- YAML-based configuration system (experiment, concept, model configs)
- Three built-in safety concepts: refusal, sycophancy, deception
- Auto-discovering data source registry for contrastive pair generation
- Dataset formatter registry (alpaca, dolly, code_alpaca)
- Behavioral evaluation framework with heuristic scorers
- Visualization module (drift plots, AUROC curves, metric heatmaps, cross-concept evolution)
- Shared utilities: model loading, chat template helpers, CSV readers
- CLI entry points: `safety-compass-extract`, `safety-compass-finetune`, `safety-compass-pairs`
- GitHub Actions CI (pytest + ruff, Python 3.9-3.12)
- Experimental results: cross-dataset fragility hierarchy (refusal > sycophancy > deception)
- Behavioral validation: drift-to-behavior correlation confirmed for refusal
