# SYNTHETIC authored prompts (not real faculty data)

`authored_prompts_SYNTHETIC.csv` holds 12 prompts written by an AI in the voices of 12 different teaching roles.
They are NOT real faculty prompts and must never be counted as such, mixed into `data/gold/inbox/`, or cited as evidence of accuracy.
The label columns are the author's intended labels, written together with the text, so they are not independent annotation.

Allowed uses: regression tests, demos, checking that the UI and scripts run.
Not allowed: extractor accuracy, preference-model training, any figure in the paper.

Label rules used by the author:
- single_correct: text says one or exactly one correct answer. multi_interp: text allows more than one correct answer.
- hidden_answers: answers withheld from the question list until later. answer_after_each_q: answer shown right after each question.
- ordered_easy_to_hard: text asks for a difficulty progression. no_basic: text excludes basic recall questions.
- Slots 0, 0.5 or 1. role needs an instruction to the AI such as "Act as". explanation is 1 if it covers right and wrong options, 0.5 if only the right one or unspecific.
