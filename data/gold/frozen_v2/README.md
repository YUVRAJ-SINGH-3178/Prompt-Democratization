# Frozen test set v2 (SYNTHETIC, authored by Claude, 30 prompts)

Not real faculty data. Authored text with intended labels, so it measures extractor behaviour on unseen wording, not accuracy on real faculty writing.
Do not read the label columns, tune on this file or copy its prompts into dev data before the single final measurement.
Hash rule: SHA-256 of the file with CRLF converted to LF. Compare with FROZEN_V2_SHA256.txt.

Label rules:
- single_correct: text says one, exactly one, only one or a single correct answer. multi_interp: text allows more than one correct answer or defensible option.
- hidden_answers: answers withheld from the question list until later (key at the end counts). answer_after_each_q: answer shown right after each question.
- ordered_easy_to_hard: text asks for easy-to-hard progression. A mix in random order, or hardest first, is 0.
- very_easy / expert_level / no_basic: stated as a requirement. A negated phrase ("not easy", "no very easy questions") does not assert it.
- confusing_language: text asks for tricky or confusing wording. "Do not use tricky wording" is 0.
- Slots 0, 0.5, 1. role needs an instruction such as Act as, You are, As a. audience 1 for an explicit level, 0.5 for just "students" or "my class".
  topic 1 specific, 0.5 subject only. count 1 explicit number (words count). n_options 1 explicit. labeling 1 explicit labels or numbering.
  marking 1 if the prompt says how or where the correct answer is shown or listed. explanation 1 for right and wrong options, 0.5 for right only or unspecific.
  ordering 1 for an explicit order or progression, 0.5 for a mention of mixed difficulty without order.
- conflict_energy: minimum number of requirements to drop so no conflict pair or policy conflict remains.
