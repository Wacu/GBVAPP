# Reviewing the validation sample

`docs/validation_sample.csv` holds 300 tweets drawn from the `cleaned` table
(seed 20261028), each pre-labelled by Claude. Your review turns it into the
reference answer that **both** classifiers are then scored against.

Regenerate it with `python tools/draw_validation_sample.py`.

## What to fill in

Three columns, at the front of the sheet:

| Column | What goes in it |
|---|---|
| `review_is_violence` | `TRUE` if the tweet describes, reports or alleges a specific act of gender-based violence against a person. `FALSE` otherwise. |
| `review_form` | One of Economic / Emotional / Physical / Sexual Violence, or Harmful Traditional Practices. Leave blank when `review_is_violence` is FALSE. |
| `review_notes` | Anything that made the call hard. These notes are the most useful output of the exercise. |

Work from the `tweet` column. The `claude_*` and `legacy_*` columns are there
so disagreements can be measured afterwards -- not so they can be confirmed.
Where either disagrees with what you read, you are right and it is wrong.

## What the pre-labels currently say

The trained model assigned one of four violence forms to **all 300** rows; it
has no other option. Claude marked **33** as violence and **267** as not.

Of the 33: Physical Violence 30, Emotional Violence 3. No Economic, Sexual or
Harmful Traditional Practices cases were found in this sample.

By text type: other 131, organisational post 86, awareness campaign 40, news
report 37, disclosure 6.

## Known weaknesses in the pre-labels -- look hardest here

These came out of a spot-check and are the places the pre-labels are least
trustworthy. They are not the only errors, just the ones already identified.

1. **News reports are treated inconsistently.** 16 of the 37 rows Claude called
   a news report are marked *not* violence, including a report naming a victim
   who survived an attack. A third-party report of a specific incident should
   be TRUE. This is a flaw in the written instructions, which do not say
   clearly enough that reporting an incident counts. Expect to overturn a
   number of these.

2. **`text_type` is less reliable than `is_violence`.** Commentary such as
   "so tired of so many men murdering women" was labelled a *disclosure*, which
   it is not. Treat the type column as unverified.

3. **The `evidence` column sometimes does not support the verdict.** One row
   justified Physical Violence with "prayers for the families and motherless
   child" -- a reasonable inference, but not evidence of an act. Reading
   evidence against the verdict is the fastest way to spot a bad row, which is
   why the column exists.

4. **Six rows are in Urdu, Telugu, Malayalam, Bengali or Hindi.** If you cannot
   read them, put that in `review_notes` and leave the labels blank rather than
   guessing. They are worth knowing about regardless: the trained model's word
   vectors are English-only, so it is blind to all of them.

## A caution about this corpus

The most frequent phrases in the `cleaned` table are "affect divorce",
"divorce florida" and "call office schedule consultation" -- a large share of
these tweets are US family-law firm marketing that carries a
`#domesticviolence` hashtag. The sample is therefore not representative of
Kenyan GBV discourse, and no accuracy figure from it should be presented as
one. It is adequate for comparing the two classifiers against each other,
which is all it is being used for.

## After the review

Run the scoring step (not yet written) to produce per-form precision and
recall for both approaches, plus the decisive figure: accuracy on the rows you
marked FALSE, where the trained model scores zero by construction.
