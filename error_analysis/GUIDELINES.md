# Manual Error-Analysis Guidelines

Use the same seeded 100 test IDs for every system. Two annotators independently label each output before discussion; then set `annotation_status` to `ADJUDICATED`, fill `adjudicated_labels`, and record the reason for disagreements.

## Labels

- `CORRECT`: semantically correct; wording may differ from the single reference. It cannot coexist with another label.
- `MISTRANSLATION`: source content is represented incorrectly, excluding the more specific categories below.
- `OMISSION`: meaningful source information is absent.
- `UNSUPPORTED_ADDITION`: information not supported by the source is introduced. Do not use this merely for a paraphrase.
- `ENTITY_ERROR`: a person, place, dynasty, title, organization, or named entity is wrong or confused.
- `NUMBER_OR_TIME_ERROR`: a quantity, date, reign year, duration, sequence, or temporal relation is wrong.
- `OTHER`: a substantive problem not covered above; explain it in notes.

Error labels may coexist. For example, an omitted reign year can receive both `OMISSION` and `NUMBER_OR_TIME_ERROR`. Prefer semantic judgment over literal reference matching.

## Procedure

1. Pilot ten rows together and resolve interpretation differences.
2. Annotate all remaining rows independently without seeing the other member’s labels.
3. Discuss disagreements and record adjudicated labels plus a short evidence-based note.
4. Run `summarize-error-analysis`; it rejects incomplete rows, unknown labels, and `CORRECT` combined with errors.
5. Inspect examples behind every reported count before copying results into the report.

