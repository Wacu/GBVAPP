"""Draw the validation sample and pre-label it with Claude.

This produces the spreadsheet a person reviews. It is step 2 of the plan in
docs/claude-classifier-spec.md, and nothing downstream is valid without the
human pass: Claude proposes, a person decides, and the reviewed column becomes
the reference answer that *both* classifiers are then scored against. Claude's
own output is never the yardstick for itself.

The sample is stratified by the form the trained model predicts, so its rarer
outputs appear rather than being swamped. That is a property of the sampling,
not an endorsement of those labels.

Usage:
    python tools/draw_validation_sample.py [--n 300] [--seed 20261028]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from apps.claude_classify import DEFAULT_MODEL, classify_batch, cost_of
from apps.gbv_models import load_models
from apps.predict import predict_batch

OUT = Path('docs/validation_sample.csv')

# Columns the reviewer fills in. They are written empty and placed first so
# they are what you land on when the file opens.
REVIEW_COLUMNS = ['review_is_violence', 'review_form', 'review_notes']


def load_rows(db='gbv.db'):
    """Read the cleaned table, keeping rows that still have text."""
    con = sqlite3.connect(db)
    try:
        data = pd.read_sql_query(
            'select tweet, date8, year, lemma_nostops from cleaned', con)
    finally:
        con.close()

    usable = data[data['lemma_nostops'].notna()]
    usable = usable[usable['lemma_nostops'].astype(str).str.strip() != '']
    return usable.reset_index(drop=True)


def legacy_labels(rows):
    """Label every row with the trained model, to stratify on."""
    svm = list(load_models().values())[1]
    return predict_batch(rows['lemma_nostops'].tolist(), svm)


def stratified_sample(rows, legacy, n, seed):
    """Take n rows, spread across the forms the trained model predicts.

    Proportional allocation would hand back almost nothing for a form the
    model rarely emits (Sexual Violence is 4 rows in 500), so each group
    contributes its proportional share or all of itself, whichever is smaller,
    and any shortfall is topped up at random from what is left.
    """
    frame = rows.copy()
    frame['legacy_form'] = legacy['form'].values
    frame['legacy_confidence'] = legacy['confidence'].values

    groups = frame.groupby('legacy_form', sort=False)
    share = max(n // max(len(groups), 1), 1)

    picked = []
    for _, group in groups:
        take = min(share, len(group))
        picked.append(group.sample(take, random_state=seed))
    sample = pd.concat(picked)

    if len(sample) < n:
        remaining = frame.drop(sample.index)
        topup = min(n - len(sample), len(remaining))
        if topup:
            sample = pd.concat([sample, remaining.sample(topup, random_state=seed)])

    # Shuffle so the reviewer does not read the file in label order, which
    # would anchor their judgement on what the model said.
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()

    rows = load_rows()
    print('Usable rows in `cleaned`: {}'.format(len(rows)))

    print('Labelling with the trained model, to stratify on...')
    legacy = legacy_labels(rows)
    print('  trained-model distribution: {}'.format(
        legacy['form'].value_counts().to_dict()))

    sample = stratified_sample(rows, legacy, args.n, args.seed)
    print('Sample drawn: {} rows (seed {})'.format(len(sample), args.seed))

    def tick(done, total):
        if done % 25 == 0 or done == total:
            print('  pre-labelled {}/{}'.format(done, total))

    print('Pre-labelling with {}...'.format(args.model))
    claude, totals = classify_batch(
        sample['tweet'].tolist(), model=args.model,
        max_workers=args.workers, progress=tick)

    out = pd.DataFrame({
        'review_is_violence': '',
        'review_form': '',
        'review_notes': '',
        'tweet': sample['tweet'],
        'year': sample['year'],
        'claude_is_violence': claude['is_violence'].values,
        'claude_form': claude['form'].values,
        'claude_text_type': claude['text_type'].values,
        'claude_sentiment': claude['sentiment'].values,
        'claude_evidence': claude['evidence'].values,
        'claude_confidence': claude['confidence'].values,
        'legacy_form': sample['legacy_form'],
        'legacy_confidence': sample['legacy_confidence'],
        'claude_error': claude['error'].values,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding='utf-8-sig')

    print()
    print('Wrote {} ({} rows)'.format(OUT, len(out)))
    print('Pre-labelling: {} calls, {} errors, cost ${:.4f}'.format(
        totals['calls'], totals['errors'], cost_of(totals, args.model)))
    print()
    print('Claude distribution:')
    print('  is_violence true : {}'.format(int(claude['is_violence'].sum())))
    print('  is_violence false: {}'.format(
        int((claude['is_violence'] == False).sum())))
    print('  forms            : {}'.format(claude['form'].value_counts().to_dict()))
    print('  text types       : {}'.format(
        claude['text_type'].value_counts().to_dict()))
    print()
    print('Next: fill review_is_violence (TRUE/FALSE) and review_form for every')
    print('row, working from the tweet and ignoring both model columns where')
    print('they disagree with what you read.')


if __name__ == '__main__':
    main()
