# Table of Contents

* [prompts](#prompts)
  * [doCachedReplacements](#prompts.doCachedReplacements)
  * [conversationToString](#prompts.conversationToString)

<a id="prompts"></a>

# prompts

<a id="prompts.doCachedReplacements"></a>

#### doCachedReplacements

```python
def doCachedReplacements(funcName, tokenizer, getMessagesFunc,
                         replacementsDict, tokenizerArgs)
```

Optimization to substantially speed up tokenization by caching the results and doing string substitutions at the end
Requires putting REPLACE at the end of each thing you replace, I did this to avoid overlaps with existing stuff in the data

<a id="prompts.conversationToString"></a>

#### conversationToString

```python
def conversationToString(conversation: List[Dict[str, str]], tokenizer,
                         maxTokens: int) -> str
```

Converts a conversation like
[
    {"role": "user", "content": "Hi there"},
    {"role": "assistant", "content": "Hi:3"}
]
into a corresponding string
User:
Hi there
Assistant:
Hi:3
And truncates (rounding down to conversation boundaries) to maxTokens

