# Auto-Test

Auto-Test is a comprehensive evaluation suite designed for production-ready AI models. It combines objective issue detection, subjective attribute evaluation, and simulated conversations to ensure both technical reliability and user satisfaction.

![image](https://github.com/fangyuan-ksgk/auto-test/assets/66006349/f75eb781-f75d-47ff-9516-14a9f9dc0847)


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
