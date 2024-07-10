# Auto-Test
Comprehensive evaluation suite for production-ready AI models.

* Objective Issue Detection: Identifies safety, coherence, and contextual problems through automated testing.
```python
python -m src.detect 
```
Note that '-m 0' detects issue for the fine-tuned model, while '-m 1' detects issue for the baseline model. 

* Subjective Assessment w. Human Supervision: Evaluates user experience and style alignment using AI-assisted tools and human feedback. This mode asks for Human acceptance of the evaluation result. All human-supervised evaluation result will be stored and used for future evaluation & alignment. 
```python
python -m src.eval -m 1
```

* Automatic Subjective Assessment. This mode performs evaluation based on historical human annotation and LLM's reasoning ability. 
```python
python -m src.eval -m 0
```
* Continuous Alignment: Stored human preference could be used to align the evaluator for better accuracy

Ensures both technical reliability and user satisfaction for deployed AI models.

![image](https://github.com/fangyuan-ksgk/auto-test/assets/66006349/0baab3ac-c0e6-4ded-937a-95060fd60aea)


Note that one would need to insert values into 'config.py', this includes vLLM served model, API keys for OPENAI, ANTHROPIC, and OPENROUTER

```shell
OPENROUTER_API_KEY
OPENAI_API_KEY
ANTHROPIC_API_KEY
```
