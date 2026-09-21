# News Verification App

Check online claims using AI-assisted evidence retrieval. Paste a claim or headline; the app
searches the web, hands the retrieved evidence to an AI model, and shows a verdict
(**SUPPORTED**, **DISPUTED** or **UNCLEAR**) with a plain-language explanation and links to the
real sources it used.

> The result is **not absolute truth**. It reflects only the sources that were retrieved.
> Always open the links and judge for yourself.

## Tech stack (all free tiers)

| Part | Choice | Why |
|---|---|---|
| UI | Streamlit | Pure Python, one-command run, free hosting on Streamlit Community Cloud |
| Search | [Tavily](https://tavily.com) Search API | Free tier (no credit card), returns titles, URLs and text snippets |
| AI | [Groq](https://console.groq.com) (Llama 3.3 70B) via an OpenAI-compatible API | Free tier, fast, supports JSON output. Can be swapped via env vars |

Only three Python packages are needed: `streamlit`, `requests`, `python-dotenv`.

## Project structure

```text
news-verification-app/
├── main.py                       # Streamlit UI + entry point
├── requirements.txt
├── .env.example                  # template for your keys
├── .gitignore
├── models/verification_result.py # dataclasses: Source, EvidenceItem, AIAnalysis, VerificationResult
├── services/
│   ├── search_service.py         # SearchProvider interface + Tavily implementation
│   ├── ai_service.py             # prompt, API call, strict validation of the AI's JSON
│   └── verification_service.py   # coordinates the whole pipeline
├── utils/helpers.py              # input validation, URL safety, source labels, JSON parsing
└── tests/test_verification.py    # automated tests (no keys or internet needed)
```

## How the components communicate

```text
main.py (UI)
   │  claim text
   ▼
VerificationService.verify()
   1. validate_claim()                      -> reject empty / too short / too long input
   2. generate_queries()                    -> 2 search queries
   3. SearchProvider.search()  ──► Tavily   -> list[Source]  (real URLs only)
   4. _select_sources()                     -> drop duplicates/empty, list better-known types first
   5. AIService.analyze()      ──► Groq     -> AIAnalysis (validated JSON)
   ▼
VerificationResult  ──►  main.py renders verdict, explanation, evidence, sources
```

Key design points:

* **Sources come only from the search service.** The AI is given numbered sources *without URLs*
  and cites them by number, so it cannot invent or alter a link. Citations to non-existent source
  numbers are discarded.
* **A firm verdict needs evidence.** If the AI says SUPPORTED/DISPUTED but cites no valid
  evidence, the result is automatically downgraded to UNCLEAR. If the search returns nothing
  usable, the AI is not called at all and the app shows UNCLEAR.
* **Credibility handling is a label, not a score.** Each source gets a category from simple rules
  (`.gov` → Government, `.edu`/`.ac.xx` → University, plus short lists of known news, fact-checking
  and scientific organizations; everything else is "Other"). Categories are shown to the user and
  used to list sources in a sensible order and to tell the AI how cautious to be. Nothing is
  hidden and no number is invented. The lists in `utils/helpers.py` are easy to edit.
* **Errors never crash the app.** Timeouts, bad keys, rate limits, unreadable responses and
  empty results all become friendly messages.

## Setup

### 1. Get the free API keys

**Search key (Tavily)**
1. Go to <https://app.tavily.com> and sign up (no credit card on the free plan).
2. Copy your API key from the dashboard (it starts with `tvly-`).

**AI key (Groq)**
1. Go to <https://console.groq.com/keys> and sign in.
2. Click **Create API Key** and copy it.

Free-tier limits and model names change over time. If something stops working, check the
providers' current docs and update `AI_MODEL` in your `.env`.

### 2. Install

Requires Python 3.9+.

```bash
cd news-verification-app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Create your `.env` file

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and fill in your keys:

```text
SEARCH_API_KEY=tvly-your-key-here
AI_API_KEY=your_groq_key_here
```

`.env` is listed in `.gitignore`, so it will not be committed. Never paste keys into source code.

### 4. Run

```bash
streamlit run main.py
```

Open <http://localhost:8501>.

## Testing

### Automated tests

```bash
python -m unittest discover -v
```

They use fake search/AI services, so they need no keys or internet. They cover:

| # | Scenario | Expected |
|---|---|---|
| 1 | Evidence supports claim | `SUPPORTED` |
| 2 | Evidence contradicts claim | `DISPUTED` |
| 3 | No usable evidence | `UNCLEAR` + "No sufficient evidence was found to confidently evaluate this claim." |
| 4 | Empty input | "Please enter a claim." |
| 5 | Search failure / timeout / bad key / rate limit | Friendly error, no crash |

Plus AI-response validation (malformed JSON, invalid verdict, invented source numbers, clamped confidence).

### Manual test with real services

Verdicts come from a live AI model and live search results, so wording can vary between runs.
Try these in the running app and inspect the sources it shows:

| Claim | Typically expected |
|---|---|
| NASA landed astronauts on the Moon in 1969. | SUPPORTED |
| 5G mobile networks spread the coronavirus. | DISPUTED |
| Scientists have discovered that drinking coffee completely prevents heart disease. | DISPUTED or UNCLEAR |
| My neighbour's cat can predict lottery numbers. | UNCLEAR |
| *(leave empty and click Verify Claim)* | "Please enter a claim." |
| *(set a wrong `SEARCH_API_KEY` and verify)* | Friendly "rejected the API key" message |

## Deploy to Streamlit Community Cloud (free)

1. Push the project to a **public GitHub repository** (make sure `.env` is not committed).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose your repository and branch, and set **Main file path** to `main.py`.
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   SEARCH_API_KEY = "tvly-your-key-here"
   AI_API_KEY = "your_groq_key_here"
   ```

   Top-level secrets are exposed to the app as environment variables, which is what the code reads.
5. Click **Deploy**. Streamlit installs `requirements.txt` automatically.

Anyone with the link can use the app and spend *your* free quota, so keep the app link
within your class or add a simple limit if you share it widely.

## Swapping providers

**Search provider.** Create a class in `services/search_service.py`:

```python
class MySearchProvider(SearchProvider):
    def search(self, query, max_results=5):
        ...  # call your API, return a list of Source(...) or raise SearchError("friendly message")
```

Then add `"mine": MySearchProvider` to the `PROVIDERS` dict and set `SEARCH_PROVIDER=mine`.
Nothing else in the app changes.

**AI provider.** `ai_service.py` speaks the OpenAI-compatible `/chat/completions` format, so
you only change `.env`. For example (check each provider's current docs for exact values):

```text
# Google Gemini (free tier)
AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
AI_MODEL=gemini-2.5-flash

# Local Ollama (fully offline, no key needed; put any text in AI_API_KEY)
AI_BASE_URL=http://localhost:11434/v1
AI_MODEL=llama3.1
AI_API_KEY=ollama
```

## Limitations (good to mention in a presentation)

* Search snippets are short excerpts, not full articles, so the AI sees limited context.
* Source labels are a simple domain-based heuristic; a well-known outlet can still be wrong and an
  unknown site can be right.
* The AI's "confidence" is self-reported, not a calibrated probability.
* Very recent events may have few sources yet, which correctly leads to UNCLEAR.
* Results can vary slightly between runs.
