"""GBV classification via the Claude API.

Unlike `apps/predict.py`, this reads the *raw* tweet. `apps/cleaning.py` strips
every non-letter character and hardcodes `sexual -> sex` and `raped -> rape`;
it removes exactly the negation, ordering and code-switching a language model
relies on. Passing it cleaned text would throw away the reason for using it.

The record it returns is deliberately wider than a form label. The trained
estimators expose `classes_ = [0 1 2 3]` and `predict` is an argmax over them,
so they cannot say "this is not about violence" -- see `apps/predict.py`.
`is_violence` is that missing field, and `evidence` makes a wrong call
auditable instead of opaque.
"""

import json
import os

import pandas as pd

# Haiku is the working default: the rubric dominates the token count, so the
# cheapest capable model is the right starting point. Escalate only if the
# validation set says so -- see docs/claude-classifier-spec.md.
DEFAULT_MODEL = 'claude-haiku-5-5'

FORMS = [
    'Economic Violence',
    'Emotional Violence',
    'Physical Violence',
    'Sexual Violence',
    'Harmful Traditional Practices',
]

TEXT_TYPES = ['disclosure', 'news report', 'awareness campaign',
              'organisational post', 'other']

RUBRIC = """You classify short social media posts for a gender-based violence
(GBV) monitoring tool used in Kenya. Posts are collected by hashtag, so most
mention GBV as a topic without describing an incident.

Decide `is_violence` first: true only when the post describes, reports or
alleges a specific act of gender-based violence against a person. It is false
for awareness campaigns, slogans, policy discussion, fundraising and sponsor
thanks, statistics, jokes, and posts that merely use a GBV hashtag.

If `is_violence` is false, set `form` to null. Do not guess a form to fill the
field.

The forms:
- Economic Violence: controlling money, property or employment -- withholding
  income, barring work, seizing land or inheritance.
- Emotional Violence: threats, humiliation, insults, intimidation, isolation,
  coercive control.
- Physical Violence: bodily harm or the threat of it -- beating, assault,
  killing.
- Sexual Violence: rape, sexual assault, coercion, harassment, defilement.
- Harmful Traditional Practices: FGM/cut, child or forced marriage, widow
  inheritance, so-called honour violence.

Where several apply, pick the most severe act actually described.

`text_type` separates an incident signal from campaign noise. A first-person or
named-victim account is a `disclosure`; a journalistic or third-party report of
an incident is a `news report`.

Posts may be in English, Swahili or Sheng, or mix them. Read the language as
written; do not assume English.

Negation and reported speech matter. "He never laid a hand on her" is not
physical violence.

`evidence` must be text copied verbatim from the post -- the words the verdict
rests on. Use null when `is_violence` is false.

`confidence` is your own: `high` when the post states the act plainly, `medium`
when it is implied, `low` when you are reading between the lines."""

SCHEMA = {
    'type': 'object',
    'properties': {
        'is_violence': {
            'type': 'boolean',
            'description': 'Does the post describe a specific act of GBV?',
        },
        # A nullable enum has to be written as anyOf -- the API rejects a
        # union `type` with an `enum` beside it ("Enum value does not match
        # declared type").
        'form': {
            'anyOf': [{'type': 'string', 'enum': FORMS}, {'type': 'null'}],
            'description': 'Null when is_violence is false.',
        },
        'text_type': {'type': 'string', 'enum': TEXT_TYPES},
        'sentiment': {
            'type': 'string',
            'enum': ['negative', 'neutral', 'positive'],
        },
        'evidence': {
            'anyOf': [{'type': 'string'}, {'type': 'null'}],
            'description': 'Verbatim span from the post, or null.',
        },
        'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
    },
    'required': ['is_violence', 'form', 'text_type', 'sentiment',
                 'evidence', 'confidence'],
    'additionalProperties': False,
}

# Columns every row carries, so a failed call still lines up with the rest.
_FIELDS = list(SCHEMA['properties'])


def build_client(api_key=None):
    """Construct the API client.

    The key is read from the environment. It is never written to the repo --
    see the .env entry in .gitignore.
    """
    import anthropic

    key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not key:
        raise RuntimeError(
            'ANTHROPIC_API_KEY is not set. Set it in the environment; do not '
            'put it in a file inside the repository.'
        )
    return anthropic.Anthropic(api_key=key)


def classify_one(text, client, model=DEFAULT_MODEL):
    """Classify a single post. Returns (record dict, usage object)."""
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=[{
            'type': 'text',
            'text': RUBRIC,
            # The rubric is identical on every call and dwarfs the post, so it
            # is the whole reason per-tweet cost stays low. Check
            # usage.cache_read_input_tokens -- if it is always 0 the rubric is
            # below this model's minimum cacheable prefix and the saving is
            # not actually happening.
            'cache_control': {'type': 'ephemeral'},
        }],
        # Classification needs no deliberation, and `low` is a real saving:
        # this model defaults to `medium`.
        output_config={
            'effort': 'low',
            'format': {'type': 'json_schema', 'schema': SCHEMA},
        },
        messages=[{'role': 'user', 'content': text}],
    )

    if response.stop_reason == 'refusal':
        category = getattr(response.stop_details, 'category', None)
        raise RuntimeError('Request declined: {}'.format(category))

    body = ''.join(b.text for b in response.content if b.type == 'text')
    return json.loads(body), response.usage


def classify_batch(texts, client=None, model=DEFAULT_MODEL):
    """Classify every post, returning one row per input.

    One call per post, so a failure costs one row rather than the batch. The
    corpus run should go through the Message Batches API instead (50% off);
    this is the interactive path.

    Returns (DataFrame, usage totals dict). A row that errored carries the
    reason in `error` and nulls elsewhere, so it stays aligned with its text.
    """
    client = client or build_client()

    rows = []
    totals = {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0,
              'calls': 0, 'errors': 0}

    for text in texts:
        row = {'text': text, 'error': None}
        try:
            record, usage = classify_one(text, client, model=model)
        except Exception as err:
            row.update({field: None for field in _FIELDS})
            row['error'] = '{}: {}'.format(type(err).__name__, err)
            totals['errors'] += 1
        else:
            row.update(record)
            totals['input'] += usage.input_tokens
            totals['output'] += usage.output_tokens
            totals['cache_read'] += getattr(usage, 'cache_read_input_tokens', 0) or 0
            totals['cache_write'] += getattr(usage, 'cache_creation_input_tokens', 0) or 0
        totals['calls'] += 1
        rows.append(row)

    return pd.DataFrame(rows, columns=['text'] + _FIELDS + ['error']), totals


# Published per-MTok rates, used to price a run. Cache reads are a tenth of the
# input rate; cache writes are 1.25x.
PRICES = {
    'claude-haiku-5-5': (0.10, 0.50),
    'claude-sonnet-5-5': (2.00, 10.00),
    'claude-opus-5-5': (4.00, 20.00),
}


def cost_of(totals, model=DEFAULT_MODEL):
    """Dollar cost of a run, from its token totals."""
    rate_in, rate_out = PRICES[model]
    return (
        totals['input'] * rate_in
        + totals['cache_read'] * rate_in * 0.10
        + totals['cache_write'] * rate_in * 1.25
        + totals['output'] * rate_out
    ) / 1_000_000
