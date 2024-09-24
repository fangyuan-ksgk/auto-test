# Auto-Test

Evaluation, Aligned with Human Preference

![Aligning Evaluator-small](https://github.com/user-attachments/assets/8c9e1dc4-a5bf-4dce-96c6-6b2ddf6794be)

## Key Features

### 1. Objective Issue Detection (`src.detect`)

Automatically identifies safety, coherence, and contextual problems through automated testing.

Usage:
```
python -m src.detect [-m MODEL_TYPE]
```
- `-m 0`: Detects issues for the fine-tuned model
- `-m 1`: Detects issues for the baseline model

### 2. Subjective Attribute Evaluation (`src.evaluate`)

Evaluates user experience and style alignment using AI-assisted tools and human feedback.

#### Human-Supervised Evaluation
```
python -m src.eval -m 1
```
This mode requires human acceptance of the evaluation results. All human-supervised evaluation results are stored and used for future evaluation and alignment.

#### Automatic Subjective Assessment
```
python -m src.eval -m 0
```
Performs evaluation based on historical human annotations and the language model's reasoning ability.

### 3. Simulated Conversation (`src.simulate`)

Simulates conversations to test the model's performance in realistic scenarios.

## Continuous Alignment

Stored human preferences are used to align the evaluator, improving accuracy over time. This ensures that the evaluation process adapts to evolving user expectations and model capabilities.

## Benefits

- Comprehensive testing: Covers both objective and subjective aspects of model performance
- Human-in-the-loop: Incorporates human feedback for more accurate and relevant evaluations
- Adaptability: Continuously improves evaluation criteria based on stored preferences
- Flexibility: Supports evaluation of both fine-tuned and baseline models

Auto-Test provides a robust framework for ensuring that deployed AI models meet high standards of technical reliability and user satisfaction.
