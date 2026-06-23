import dspy
from pathlib import Path
import requests
import json

'''
process haiku_examples
'''
examples = []
with open("haiku_examples.jsonl") as f:
    for line in f:
        row = json.loads(line)
        examples.append(
            dspy.Example(
                location=row["location"],
                season=row["season"],
                mood=row["mood"],
            ).with_inputs("location", "season", "mood")
        ) 

n = len(examples)
train_end = int(n * 0.75)
val_end = int(n * 0.875)
train, val, test = examples[:train_end], examples[train_end:val_end], examples[val_end:]
# Debug 模式：先只跑少量样本，确认 GEPA 流程能跑通
DEBUG = True

if DEBUG:
    train = train[:4]
    val = val[:2]
    test = test[:2]

print(f"Using train={len(train)}, val={len(val)}, test={len(test)}")

'''
evaluate metrics
'''
def haiku_score(example, prediction) -> float:
    """
    Penalize verbatim use of the input season string.
    A haiku should evoke the season through imagery, not name it
    directly.
    """
    text = prediction.haiku.lower()
    if example.season.strip().lower() in text:
        return 0.0
    return 1.0

'''
API key
'''
api_key_file = Path(__file__).resolve().with_name("apikey.txt")
DS_API_KEY = api_key_file.read_text(encoding="utf-8").strip()
if not DS_API_KEY:
    raise ValueError(f"API key file is empty: {api_key_file}")

'''
for wikiapi request
'''
USER_AGENT = "LocalDSPyReActDemo/0.1 (your_email@example.com)"
WIKI_API = "https://en.wikipedia.org/w/api.php"

'''
initial dspy model
'''
# Pass the key explicitly...
lm = dspy.LM("deepseek/deepseek-v4-flash", api_key=DS_API_KEY)
dspy.configure(lm=lm)

'''
wikipedia-based tools to use by dspy.ReAct
'''
def wikipedia_search(query: str, limit: int = 5) -> list[str]:
    """Search English Wikipedia for a query and return candidate page titles."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }

    resp = requests.get(
        WIKI_API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()

    data = resp.json()
    return [item["title"] for item in data["query"]["search"]]


def get_wikipedia_page(title: str) -> str:
    """Fetch the plain-text extract of an English Wikipedia page by title."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }

    resp = requests.get(
        WIKI_API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()

    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        return f"No Wikipedia page found for title: {title}"

    return page.get("extract", "")[:6000]


'''
class based signatures
'''
from typing import Literal

Season = Literal[
    "spring", "summer", "autumn", "winter",
]

class HaikuBot(dspy.Signature):
    """
    Write a classical haiku given the provided inputs.
    """
    location: str = dspy.InputField()
    mood: str = dspy.InputField()
    season: Season = dspy.InputField()
    haiku: str = dspy.OutputField()

haiku_bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page], max_iters=4)

from haiku_metric import haiku_metric

reflection_lm = dspy.LM("deepseek/deepseek-v4-flash", api_key=DS_API_KEY)

'''
optimizer = dspy.GEPA(
    metric=haiku_metric,
    reflection_lm=reflection_lm,
    auto="light",
    num_threads=2,
)
'''

optimizer = dspy.GEPA(
    metric=haiku_metric,
    reflection_lm=reflection_lm,
    max_metric_calls=12,
    num_threads=2,
    reflection_minibatch_size=1,
    use_merge=False,
)

optimized_haiku_bot = optimizer.compile(haiku_bot, trainset=train, valset=val)
optimized_haiku_bot.save("react_gpt_nano_haiku_optimized.json")