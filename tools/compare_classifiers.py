"""Run the same texts through the trained models and through Claude.

The point is not that one scores higher on a toy set -- 18 hand-written texts
prove nothing statistically. The point is to show *where* and *why* the two
disagree, using cases chosen to probe known structural limits of the trained
pipeline: no negative class, summed GloVe vectors (so word order and negation
are discarded), and an English-only vocabulary.

A proper comparison needs the ~300 hand-reviewed rows described in
docs/claude-classifier-spec.md. This is the demonstration, not the evaluation.

Usage:
    ANTHROPIC_API_KEY=... python tools/compare_classifiers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from apps.claude_classify import DEFAULT_MODEL, classify_batch, cost_of
from apps.cleaning import lem, text_cleaning
from apps.gbv_models import load_models
from apps.predict import predict_batch

# Each case names the limit it probes, so a reader can tell a real failure from
# a lucky guess. `expect` is the honest answer; None means "not violence".
CASES = [
    # --- the four forms the models were actually trained on -----------------
    ('He controls every shilling I earn and I must ask him for bus fare.',
     'Economic Violence', 'trained form'),
    ('She calls me worthless in front of the children every single day.',
     'Emotional Violence', 'trained form'),
    ('My husband beat me again last night. I have nowhere to go.',
     'Physical Violence', 'trained form'),
    ('He forced himself on me after I said no.',
     'Sexual Violence', 'trained form'),

    # --- the fifth form the README documents but nothing was trained on -----
    ('My aunt is planning to have my 9-year-old sister cut next month.',
     'Harmful Traditional Practices', 'untrained form (HTP)'),
    ('They married her off at 14 to settle a family debt.',
     'Harmful Traditional Practices', 'untrained form (HTP)'),

    # --- negation: summed GloVe vectors discard it --------------------------
    ('He has never once raised a hand to her, whatever the rumours say.',
     None, 'negation'),
    ('No woman in this family has ever been beaten and we will keep it that way.',
     None, 'negation'),

    # --- Swahili / Sheng: absent from the GloVe vocabulary ------------------
    ('Amenipiga tena usiku wa leo. Sijui niende wapi.',
     'Physical Violence', 'Swahili'),
    ('Mze wangu ananitishia kunichapa kila siku, hii maisha ni ngumu.',
     'Emotional Violence', 'Sheng / code-mixed'),

    # --- several forms in one text ------------------------------------------
    ('He took my phone, locked me in, and hit me when I asked to leave.',
     'Physical Violence', 'multi-form'),
    ('After he raped me he said no one would believe a woman like me.',
     'Sexual Violence', 'multi-form'),

    # --- not violence: the class the trained models cannot express ----------
    ('Huge thanks to our sponsors for supporting the 16 Days of Activism campaign!',
     None, 'control: sponsor post'),
    ('One in three women worldwide experiences physical or sexual violence.',
     None, 'control: statistic'),
    ('Join us Saturday at Uhuru Park to march against gender-based violence.',
     None, 'control: awareness'),
    ('The new bill on domestic violence goes to a second reading on Tuesday.',
     None, 'control: policy'),
    ('My sister just got promoted to regional manager. So proud of her!',
     None, 'control: unrelated'),

    # --- third-party reporting rather than disclosure ------------------------
    ('Police in Nakuru have arrested a man accused of assaulting his wife.',
     'Physical Violence', 'news report'),
]


def legacy_predictions(texts):
    """Run the texts through the app's own pipeline: clean, lemmatise, predict."""
    cleaned = [lem(text_cleaning(t)) for t in texts]
    models = load_models()
    # The page falls back to this estimator when no model is chosen.
    svm = list(models.values())[1]
    return predict_batch(cleaned, svm)


# Texts with no GloVe vocabulary overlap at all. If the trained model returns a
# confident form for keyboard mash, then a "correct" answer on Swahili is not a
# prediction -- it is the same default landing on the right label by accident.
OOV_PROBES = [
    'qwertyuiop asdfghjkl zxcvbnm',
    'nkwelfj pqoritz mbxvucd',
    'Nimefurahi sana leo ni siku njema ya kupumzika',  # "I am very happy today"
]


def oov_check():
    """Show what the trained model does with text it has no vectors for."""
    cleaned = [lem(text_cleaning(t)) for t in OOV_PROBES]
    svm = list(load_models().values())[1]
    out = predict_batch(cleaned, svm)

    print()
    print('Out-of-vocabulary probe (trained model only):')
    for text, form, conf in zip(OOV_PROBES, out['form'], out['confidence']):
        print('  {:<50} -> {:<20} {}%'.format(text[:48], form, conf))


def main():
    texts = [c[0] for c in CASES]
    expected = [c[1] for c in CASES]
    probes = [c[2] for c in CASES]

    print('Running the trained pipeline (GloVe + SVM)...')
    legacy = legacy_predictions(texts)

    print('Running Claude ({})...'.format(DEFAULT_MODEL))
    claude, totals = classify_batch(texts)

    claude_form = [
        None if row.is_violence is False else row.form
        for row in claude.itertuples()
    ]

    out = pd.DataFrame({
        'probe': probes,
        'text': texts,
        'expected': expected,
        'legacy_form': legacy['form'],
        'legacy_conf': legacy['confidence'],
        'claude_form': claude_form,
        'claude_conf': claude['confidence'],
        'claude_type': claude['text_type'],
        'claude_evidence': claude['evidence'],
        'error': claude['error'],
    })

    legacy_right = sum(a == b for a, b in zip(out['legacy_form'], expected))
    claude_right = sum(a == b for a, b in zip(claude_form, expected))

    # The decisive subset: texts that describe no violence at all.
    controls = [i for i, e in enumerate(expected) if e is None]
    legacy_controls = sum(out['legacy_form'][i] is None for i in controls)
    claude_controls = sum(claude_form[i] is None for i in controls)

    print()
    print('=' * 100)
    print('{:<28} {:>12} {:>12}'.format('', 'GloVe+SVM', 'Claude'))
    print('{:<28} {:>12} {:>12}'.format(
        'correct overall', '{}/{}'.format(legacy_right, len(CASES)),
        '{}/{}'.format(claude_right, len(CASES))))
    print('{:<28} {:>12} {:>12}'.format(
        'correct on non-violence', '{}/{}'.format(legacy_controls, len(controls)),
        '{}/{}'.format(claude_controls, len(controls))))
    print('=' * 100)
    print()
    print('Claude run: {} calls, {} errors, {} input / {} output tokens, '
          '{} cached reads'.format(totals['calls'], totals['errors'],
                                   totals['input'], totals['output'],
                                   totals['cache_read']))
    print('Claude run cost: ${:.4f}  (${:.6f} per text)'.format(
        cost_of(totals), cost_of(totals) / max(totals['calls'], 1)))

    out.to_csv('docs/comparison_results.csv', index=False)
    print('\nFull results written to docs/comparison_results.csv')

    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 38)
    print()
    print(out[['probe', 'text', 'expected', 'legacy_form', 'legacy_conf',
               'claude_form', 'claude_conf']].to_string())

    oov_check()


if __name__ == '__main__':
    main()
