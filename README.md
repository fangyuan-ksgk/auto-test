# Auto-Test
Comprehensive evaluation suite for production-ready AI models.

* Objective Issue Detection: Identifies safety, coherence, and contextual problems through automated testing.
```python
python -m src.detect
```
* Subjective Assessment w. Human Supervision: Evaluates user experience and style alignment using AI-assisted tools and human feedback.
```python
python -m src.eval -m 1
```
This mode asks for Human acceptance of the evaluation result. All human-supervised evaluation result will be stored and used for future evaluation & alignment. 

* Automatic Subjective Assessment
```python
python -m src.eval -m 0
```
This mode automatically performs evaluation based on historical human annotation and LLM's reasoning ability.

* Continuous Alignment: Stored human preference could be used to align the evaluator for better accuracy

Ensures both technical reliability and user satisfaction for deployed AI models.

![image](https://github.com/fangyuan-ksgk/auto-test/assets/66006349/0baab3ac-c0e6-4ded-937a-95060fd60aea)


Requireed API KEYs in the environment variable

```shell
OPENROUTER_API_KEY
OPENAI_API_KEY
GROQ_API_KEY
ANTHROPIC_API_KEY
GROQ_API_KEY
```
