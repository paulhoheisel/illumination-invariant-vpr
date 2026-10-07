# Illumination-Invariant Visual Place Recognition - Individual Research Project completed within the COLABS program of the Tohoku University (Sendai, Japan)

This project explores a core computer vision problem which is for example relevant for autonomous vehicles or robots: how can a visual place recognition (VPR) model reliably match the same location across large illumination changes, especially from day to night?

I built and evaluated a research pipeline for learning illumination-invariant embeddings that remain stable when scene appearance changes dramatically. The work is designed around the real-world challenge of navigating using camera data only, where lighting variation can make traditional image descriptors fail.

## Research problem

Visual place recognition is often the first step in localization and loop closure for autonomous systems. In practice, the same place can look very different across time of day, weather, and seasonal conditions. A model that relies too heavily on raw brightness and texture cues may incorrectly match different areas or fail to recognize the correct one altogether.

This project tackles that challenge by studying how to learn embeddings that are robust to illumination shifts while preserving discriminative power for different locations.

## What I implemented

- Pose-aware preprocessing of Oxford RobotCar data
- Night/day pair construction for cross-condition matching
- Clustering and subroute-based label generation for structured training
- Margin-based metric learning for class separation and retrieval robustness
- Distillation-style alignment between day and night representations
- Evaluation scripts for illumination-invariance and threshold-based matching

## Method overview

The pipeline combines several ideas:

1. Data preparation and pose extraction
   - Camera poses are recovered from scene metadata and mapped to geometric scene locations.
   - Day and night images are paired using spatial and rotational constraints.

2. Structured training labels
   - The code groups locations into clusters and constructs subroutes to provide consistent identity labels across different conditions.
   - This helps the model learn place identity instead of simply memorizing visual appearance.

3. Illumination-robust embeddings
   - The training utilities define custom loss functions for large-margin classification and knowledge transfer between day and night branches.
   - The aim is to bring embeddings for the same place closer together, even when the visual input is significantly altered by lighting.

4. Retrieval and evaluation
   - The repository evaluates the learned embeddings using retrieval metrics and similarity tests under day/night conditions.
   - This is useful for understanding not just training behavior but also real-world place-matching performance.

## Key files

- `oxford_data_processing.py` — dataset preparation, pose handling, pair generation, clustering logic, and evaluation utilities
- `training_tools.py` — custom loss functions for large-margin classification and illumination-aware alignment
- `pipeline_Oxford.ipynb` — end-to-end Oxford-based experiment flow
- `pipelineNocPlace.ipynb` — related exploration around place recognition pipelines and training design
- `vpr-results.ipynb` — results visualization and metric analysis
- `tests/test_threshold_dataset.py` — validation for dataset grouping and invariance checks

## Technical stack

- Python
- PyTorch
- NumPy / SciPy
- scikit-learn
- Jupyter notebooks

## Why this project matters

This work sits at the intersection of robotics, representation learning, and perception. In real-world robotic deployment, a perception system must work under changing light, shadows, glare, and weather. Improving illumination invariance improves localization reliability, loop closure, and long-term autonomy.

For recruiters, this project demonstrates:

- research-oriented problem framing
- end-to-end machine learning pipeline design
- data engineering for robotics datasets
- metric learning and representation alignment
- experimentation, evaluation, and iterative model improvement

## Validation

The repository includes a lightweight test suite covering threshold-based dataset grouping and illumination-invariance evaluation. The project is intended as a research prototype and training/evaluation workspace rather than a production package.

## Takeaway

This project is a strong example of applied research in visual place recognition: I focused on the difficult but important problem of consistently recognizing locations under changing illumination, using data preprocessing, metric learning, and evaluation in a realistic robotics setting.

If you are reviewing this repository as a portfolio artifact, it highlights my ability to bring together:

- machine learning research
- computer vision for robotics
- dataset curation and preprocessing
- experimentation and analysis
- clear technical communication through reproducible code
