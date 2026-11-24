# Gold prompt inbox

Put REAL prompts here, one row each in `prompts.csv`. Real means written by a faculty member or staff member for
their own teaching, collected with their permission. Do not paste prompts written by an AI or by the builder.

Columns of `prompts.csv`:
- prompt_id: unique, for example P001
- faculty_pseudonym: a code such as F07, never a name
- department: free text
- course_topic: free text
- collected_on: YYYY-MM-DD
- consent: must be `yes`
- prompt_text: the prompt exactly as the faculty member wrote it

Annotation goes in `annotations_<annotator>.csv`, one file per annotator, annotators working separately:
- prompt_id
- one column per atom: very_easy, expert_level, no_basic, ordered_easy_to_hard, hidden_answers,
  answer_after_each_q, single_correct, multi_interp, confusing_language (value 1 if the prompt asserts it, else 0)
- one column per slot: role, audience, topic, count, n_options, labeling, marking, explanation, ordering
  (value 0, 0.5 or 1)

Rules: do not open the other annotator's file before finishing yours. Do not look at linter output while annotating.
Check the folder with: python scripts/validate_gold.py
