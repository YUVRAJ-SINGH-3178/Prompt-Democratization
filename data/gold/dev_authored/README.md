# Development Dataset: Authored Prompts

> **PROVENANCE LABEL**: `source: authored_by_builder`, `synthetic: 1`
> This development dataset was authored by the builder for tuning the rule-based requirement extractor.
> It is strictly separate from the frozen test set and any real human data.

Total prompts in dev set: 68

## Voice Distribution

- Terse: 5 prompts
- Verbose: 5 prompts
- Non-native English phrasing: 8 prompts
- Bullet lists: 8 prompts
- Numbered rules: 8 prompts
- Negations (e.g., 'do not show answers', 'avoid basic'): 8 prompts
- Paraphrases (e.g., 'strictly one right answer', 'simplest to hardest'): 8 prompts
- Hidden conflicts across separate sentences: 8 prompts
- Zero atoms (standard prompts with no special atoms): 10 prompts

## Atom Labelling Rules

1. `very_easy`: Questions must be very easy, beginner-level, or elementary.
2. `expert_level`: Questions must assess advanced research knowledge or expert-level understanding.
3. `no_basic`: Instruction explicitly excluding, skipping, or avoiding basic/introductory questions.
4. `hidden_answers`: The correct answer must be concealed or withheld from students.
5. `answer_after_each_q`: The answer must be given immediately after each question.
6. `single_correct`: Requirement that exactly one option is the correct answer.
7. `multi_interp`: Requirement allowing multiple interpretations or ambiguity in choices.
8. `clear_unambiguous`: Requirement that questions and wording must be clear and unambiguous.
9. `confusing_language`: Deliberate instruction to use confusing language or tricky statements.
10. `ordered_easy_to_hard`: Ordering questions from easiest to hardest or by increasing difficulty.

## Conflict Pairs

- `very_easy` conflicts with `expert_level`
- `very_easy` conflicts with `no_basic`
- `hidden_answers` conflicts with `answer_after_each_q`
- `single_correct` conflicts with `multi_interp`
- Policy conflicts: `clear_unambiguous` conflicts with `confusing_language` and `multi_interp`

---

# Nine Governed Prompt Slot Rubrics

Every slot receives a score of `0.0`, `0.5`, or `1.0`.

| Slot | 1.0 (Full Credit) | 0.5 (Partial Credit) | 0.0 (No Credit) |
|---|---|---|---|
| **role** (Educator Persona / Role) | Explicit role/persona framing instruction (e.g., 'Act as a university teacher', 'Role: professor', 'experienced instructor'). | Educator or teaching role mentioned without persona framing (e.g., 'for the teacher', 'as an educator'). | No persona or educator role mentioned. |
| **audience** (Target Audience / Cohort) | Specific educational cohort named (e.g., 'bachelor students', 'undergraduate', 'postgraduate', 'high school', 'freshman'). | Generic audience framing without specific cohort (e.g., 'for students', 'aimed at learners', 'suitable for a class'). | No audience specified, or students mentioned only incidentally (e.g., 'to confuse students'). |
| **topic** (Academic Subject / Topic) | Specific subject topic or identified course (e.g., 'on Normalization', 'about B-trees', 'operating systems deadlocks'). | Broad academic discipline without a specific topic (e.g., 'on computer science', 'in biology'). | No topic or discipline specified, or generic phrases ('cover important concepts'). |
| **count** (Question Quantity) | Specific exact count with number word or digits (e.g., '10 questions', '20 MCQs', 'five questions', '15 items'). | Approximate or bounded count ('around 10 questions', 'between 5 and 10 questions', 'at least 5 questions'). | Non-specific quantity ('some questions', 'a set of MCQs', 'several questions'). |
| **n_options** (Number of Options per Question) | Explicit number of options specified (e.g., 'four options', '4 choices', 'five alternatives', '3 options'). | Generic mention of choices/options without count ('multiple choices', 'options for each question'). | No mention of choices or options. |
| **labeling** (Option Labeling Scheme) | Explicit labeling scheme specified (e.g., 'labelled A, B, C, D', '(A, B, C, D)', 'options A to D'). | Generic instruction to label options without scheme ('label each choice', 'use letters'). | No labeling scheme mentioned. |
| **marking** (Correct Answer Marking) | Explicit instruction to mark, highlight, or supply the correct answer (e.g., 'mark the right answer', 'highlight correct answer in bold', 'provide the correct answer'). | Vague mention of answers ('include answers', 'with answers'). | No indication of marking or providing correct answers (or purely negative: 'do not mark answers'). |
| **explanation** (Answer Explanation / Rationale) | Mandatory explanation required (e.g., 'provide detailed explanations', 'explain why each option is correct or incorrect'). | Conditional or optional explanations ('explain wherever necessary', 'explanations optional', 'if needed'). | No explanation mentioned. |
| **ordering** (Question Ordering / Arrangement) | Specific pedagogical or difficulty ordering (e.g., 'in order of increasing difficulty', 'from easiest to hardest'). | Generic ordering instruction without difficulty criterion ('arrange them properly', 'order logically'). | No ordering instruction, or random/shuffled order. |

## Examples by Slot

### `role`: Educator Persona / Role
- **1.0**: "Act as an experienced professor of computer science."
- **0.5**: "Write questions that an instructor can use in class."
- **0.0**: "Generate multiple choice questions about sorting algorithms."

### `audience`: Target Audience / Cohort
- **1.0**: "Create questions suitable for second-year bachelor students."
- **0.5**: "Make the questions appropriate for students taking an exam."
- **0.0**: "Generate tricky questions to confuse students."

### `topic`: Academic Subject / Topic
- **1.0**: "Prepare questions on Database Normalization and functional dependencies."
- **0.5**: "Prepare multiple choice questions on computer science."
- **0.0**: "Make sure the questions cover all the important concepts."

### `count`: Question Quantity
- **1.0**: "Create exactly 15 multiple-choice questions."
- **0.5**: "Create around 10 to 12 questions."
- **0.0**: "Create some multiple-choice questions for the quiz."

### `n_options`: Number of Options per Question
- **1.0**: "Provide four choices for every question."
- **0.5**: "Each question should have multiple choices."
- **0.0**: "Write questions with answer keys."

### `labeling`: Option Labeling Scheme
- **1.0**: "Label each answer choice (A, B, C, D)."
- **0.5**: "Make sure each option has a clear label."
- **0.0**: "Provide four choices for each question."

### `marking`: Correct Answer Marking
- **1.0**: "Highlight the correct answer in bold after each question."
- **0.5**: "Include answers with the questions."
- **0.0**: "Write multiple choice questions for a classroom quiz."

### `explanation`: Answer Explanation / Rationale
- **1.0**: "Provide an explanation for why the chosen answer is correct."
- **0.5**: "Explain the answers wherever necessary."
- **0.0**: "Provide four choices and mark the right answer."

### `ordering`: Question Ordering / Arrangement
- **1.0**: "Organize the questions in order of increasing difficulty from easiest to hardest."
- **0.5**: "Arrange the questions properly in a logical sequence."
- **0.0**: "Write 10 questions on software testing."
