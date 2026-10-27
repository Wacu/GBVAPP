# Spec: Claude-based GBV classification

Status: proposed, not built. Branch `claude-classifier`.

This describes what gets built, how it will be judged, and what deliberately
does not change. It is a plan to be argued with, not a decision already taken.

## Why

The current classifier cannot say "this is not about violence". Both trained
estimators expose `classes_ = [0 1 2 3]` and `predict` is an argmax over them,
so **every** text is assigned one of the four forms. A sponsor thank-you tweet
comes back as Economic Violence at 76% confidence. Median confidence across the
`cleaned` table is 48%, and a low score means "cannot tell which form", not
"not violence" -- so a confidence threshold cannot rescue it either.

Two further limits are structural, not tuning problems:

- `FunctionText2Vec` sums GloVe vectors, so word order is discarded and the
  vector magnitude scales with text length. "He did not hit her" and "He hit
  her" are near-identical inputs.
- GloVe is English-only. A substantial share of Kenyan GBV discourse is Swahili
  or Sheng, and those tokens are simply absent from the vocabulary.

Retraining would need labelled negatives and a multilingual embedding. That is
a larger project than switching the inference call.

## What gets built

One function, `apps/claude_classify.py`, taking raw tweet text -- **not** the
output of `apps/cleaning.py`. The cleaning pipeline strips every non-letter
character and hardcodes `sexual -> sex`, `raped -> rape`; it destroys exactly
the context a language model uses.

It returns one structured record per tweet:

| field | values | why |
|---|---|---|
| `is_violence` | bool | The field the current models cannot express. |
| `form` | the four trained forms, `Harmful Traditional Practices`, or null | Null when `is_violence` is false. HTP is documented in the README but was never trainable; Claude can return it. |
| `text_type` | disclosure / news report / awareness campaign / organisational post / other | Separates an incident signal from a hashtag campaign. The map currently conflates them. |
| `sentiment` | negative / neutral / positive | Replaces VADER, which is English-only and misses negation. |
| `evidence` | verbatim span, or null | The words the verdict rests on. Makes a wrong call auditable instead of opaque. |
| `confidence` | low / medium / high | Coarse on purpose. A calibrated percentage would be a false claim. |

Enforced with the Messages API `output_config.format` so the shape is
guaranteed rather than parsed out of prose.

## Model and cost

Start on **`claude-haiku-5-5`**. The rubric is cached (~90% off repeat reads)
and the whole corpus runs through the Message Batches API (50% off).

| | per 1,000 tweets | full 567,123-row corpus |
|---|---|---|
| Haiku 5.5, cached + batched | ~$0.04 | **~$23** |
| Sonnet 5.5, cached + batched | ~$0.81 | ~$460 |

Escalate to Sonnet only if Haiku fails the validation gate below. The 500-row
`cleaned` table costs about four cents either way, so development is free in
practice; the corpus-wide number is what matters for a decision.

## How it gets judged

No rewiring until this is done, because without it we would be swapping an
unmeasured classifier for an unmeasured classifier.

1. Draw ~300 rows at random from `cleaned`, stratified across the four
   predicted forms so rare classes appear (Sexual Violence is 4 of 500).
2. Claude proposes labels. **A person reviews every one.** The reviewed set is
   the ground truth; Claude's own output is never the yardstick for itself.
3. Score both approaches on the same 300 rows: per-form precision and recall,
   and -- the decisive measure -- accuracy on the non-violence rows, where the
   current models score zero by construction.
4. Also report the disagreements. Where the two differ, which is right?

Ship only if Claude beats the current models on the forms **and** usefully
separates non-violence. If it ties on forms but gets non-violence right, that
alone justifies it.

## What does not change

- **Geocoding stays on ArcGIS.** Claude must not generate coordinates. A GBV
  incidence map with plausible, wrong locations is worse than no map, because
  people would act on it.
- **Word frequency, n-grams, wordclouds stay as they are.** Deterministic
  counting; a model would be slower, dearer and less accurate.
- **`apps/cleaning.py` stays.** The Claude path bypasses it, but the EDA views
  still need the processed text.
- **The legacy models stay selectable.** `MODEL_OPTIONS` gains a Claude entry
  rather than losing the others, so the comparison stays reproducible and the
  app still works with no API key and no network.
- **Twitter ingestion is out of scope.** It is broken because API v1.1
  `search_tweets` lost free access in 2023. That needs an API v2 rewrite; no
  model choice affects it.

## Open questions, for a person not a model

1. **Is sending tweet text to the Anthropic API cleared?** The tweets are
   public, but they are GBV-related and some are first-person disclosures.
   This needs a yes from whoever owns data governance before the corpus run.
2. **What happens when a real disclosure is detected?** The app currently has
   no answer. Adding a field that reliably flags disclosures makes that gap
   urgent rather than theoretical: a detection with no referral path is a
   liability, not a feature.
