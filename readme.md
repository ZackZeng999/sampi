# SAM-Guided OpenPI for LIBERO

This project explores how segmentation-level visual priors from SAM3 can be integrated with an OpenPI / pi0.5 vision-language-action policy for robotic manipulation in the LIBERO benchmark.

The core idea is to use SAM3 to identify task-relevant objects, dim the visual background, and adapt the OpenPI action expert to this SAM-enhanced visual distribution through LoRA fine-tuning.

## Overview

Vision-language-action models such as OpenPI can generate robot actions from images, robot states, and language instructions. However, manipulation scenes often contain distractors, visually similar objects, and irrelevant background regions. This project investigates whether object-level segmentation can make the visual input more focused and improve policy behavior.

The current system uses SAM3 as an external visual preprocessing module rather than modifying the OpenPI architecture directly.

```text
LIBERO observation
-> task-relevant prompt extraction
-> SAM3 object segmentation
-> dim-background image transformation
-> OpenPI policy inference
-> action expert LoRA fine-tuning on SAM-dimmed data
```

## Method

The method has two modes.

Online inference:

- A VLM/SAM-agent extracts important object prompts from the task description and current image.
- SAM3 predicts masks for the selected object prompts.
- The foreground object regions remain unchanged while the background is darkened.
- The processed image is sent to the OpenPI policy server for action generation.

Offline fine-tuning:

- LIBERO observations are converted into SAM-dimmed images using cached SAM masks.
- The resulting dataset preserves robot states, actions, task labels, and wrist images.
- Only the OpenPI action expert is fine-tuned with LoRA.
- The PaliGemma visual-language backbone remains frozen.

## Main Components

```text
openpi/
```

OpenPI policy code, LIBERO evaluation code, fine-tuning configuration, and dataset preprocessing scripts.

```text
sam3/
```

SAM3 model code and the local SAM agent server used by OpenPI.

```text
eval/
```

Evaluation logs for baseline, SAM-inference, and fine-tuned policy runs.

```text
thesis_notes.md
```

Project memory for thesis writing, experiment tracking, implementation details, and current results.

## Implemented Features

- SAM3 server with `/segment` and `/extract` endpoints.
- OpenPI-side SAM client for prompt extraction, mask retrieval, and dim-background preprocessing.
- LIBERO evaluation integration with optional SAM preprocessing.
- Offline prompt extraction cache for LIBERO episodes.
- Per-episode SAM mask cache.
- SAM-dimmed LeRobot dataset generation.
- OpenPI pi0.5 action-expert LoRA fine-tuning config.
- Debug and full-data fine-tuned checkpoints.
- Evaluation logging for extract time, segmentation time, OpenPI inference time, and rollout success.

## Dataset and Training Status

The current full SAM-dimmed LIBERO dataset contains:

```text
1693 episodes
273465 frames
40 tasks
```

The current fine-tuning setup:

- Base model: OpenPI pi0.5 LIBERO checkpoint.
- Trainable module: action expert LoRA.
- Frozen module: PaliGemma visual-language backbone.
- Training data: SAM-dimmed LIBERO dataset.
- First formal fine-tuning run: approximately one epoch.

## Experimental Plan

The intended comparison groups are:

- Original OpenPI pi0.5 LIBERO policy without SAM.
- Original OpenPI pi0.5 LIBERO policy with SAM-dim preprocessing at inference time.
- OpenPI pi0.5 action expert LoRA fine-tuned on SAM-dimmed LIBERO data.
- Additional epoch or backbone-adaptation ablations if needed.

Main evaluation metric:

```text
LIBERO rollout success rate
```

Additional analysis:

- Prompt extraction quality.
- SAM mask quality.
- Action inference latency.
- SAM preprocessing latency.
- Failure cases by task type.

## Research Context

This project is being developed as part of a graduation thesis on improving VLA robotic manipulation policies with segmentation-based visual enhancement and action expert adaptation.

The main research question is:

Can SAM-guided object-level visual preprocessing help a VLA policy focus on task-relevant regions and improve manipulation performance after lightweight action-expert fine-tuning?
