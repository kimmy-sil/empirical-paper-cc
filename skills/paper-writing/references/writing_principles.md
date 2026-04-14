# Writing Principles for Empirical Papers

> **Reference file** — read on demand when drafting or revising.
> Extracted and condensed from community best practices (K-Dense, Cochrane, McCloskey).

---

## Three Pillars

### Clarity
- One main idea per paragraph. Topic sentence first, then evidence, then transition.
- Use precise, unambiguous language. Replace "quite a few" with "68% (32/47)."
- Define technical terms at first use. Use abbreviations only if the term appears 3+ times.
- Prefer active voice in Results/Discussion. Passive is acceptable in Methods when the action matters more than the actor.
- Break up sentences over 30 words. Average sentence length: 15–20 words.

### Conciseness
- Every unnecessary word is a missed opportunity for clarity.
- Target: can ANY word be removed without losing meaning? If yes, remove it.
- Acceptable length: detailed method descriptions that need every element are fine.
- Not acceptable: throat-clearing phrases, redundant qualifiers, noun+verb where a verb suffices.

### Accuracy
- Report exact values with appropriate precision. Match decimal places to measurement capability.
- Distinguish observations from interpretations ("decreased from 145 to 132 mmHg" vs. "suggests effective treatment").
- Use "statistically significant" only with a p-value. Never "highly significant" for p=0.03.
- Verify all numbers: text matches tables, n values sum correctly, percentages are correct.

---

## Wordy → Concise Replacement Table

| ❌ Wordy | ✅ Concise |
|----------|-----------|
| due to the fact that | because |
| in order to | to |
| it is important to note that | *(delete)* |
| a total of 50 participants | 50 participants |
| at the present time | now / currently |
| conduct an investigation into | investigate |
| give consideration to | consider |
| has been shown to be | is |
| in the event that | if |
| make a decision | decide |
| perform an analysis | analyze |
| provide information about | inform |
| in spite of the fact that | although |
| there is evidence suggesting that | evidence suggests |
| play an important role in | affect / influence |
| with respect to | regarding / on |

---

## Verb Tense by Section

| Section | Sub-part | Tense | Example |
|---------|----------|-------|---------|
| Abstract | Background | Present / present perfect | "Depression affects..." / "Studies have shown..." |
| Abstract | Methods | Past | "We recruited 100 participants" |
| Abstract | Results | Past | "The intervention reduced symptoms" |
| Abstract | Conclusion | Present | "These findings suggest..." |
| Introduction | Established facts | Present | "Trade openness is associated with..." |
| Introduction | Prior studies | Past / present perfect | "Smith (2020) found..." / "Recent work has shown..." |
| Introduction | This paper | Past or present | "We investigate..." / "This paper examines..." |
| Methods | Your procedures | Past | "We collected data from..." |
| Results | Your findings | Past | "The coefficient was 0.05" |
| Discussion | Your findings | Past | "We found that..." |
| Discussion | Interpretation | Present | "This suggests that..." |
| Discussion | General truths | Present | "Firms respond to incentives by..." |

---

## Hedging: Getting It Right

**Too much** (sounds uncertain):
> "It could perhaps be possible that the intervention might have some effect."

**Too little** (overstates):
> "The intervention cures depression."

**Just right** (appropriate for quasi-experimental):
> "The intervention significantly reduced depressive symptoms, suggesting it may be effective for mild to moderate cases."

### Hedging Words by Certainty Level

| Level | Words | Use when |
|-------|-------|----------|
| Strong | demonstrates, establishes, confirms | RCT with large N, replicated findings |
| Moderate | indicates, shows, reveals | Quasi-experimental, consistent with theory |
| Cautious | suggests, is consistent with, points to | Single study, potential confounds remain |
| Weak | may, might, could, appears to | Exploratory, small N, descriptive |

---

## Common Pitfalls in Empirical Papers

### Ambiguous Pronouns
> ❌ "Treated firms and control firms were compared. They showed improvement."
> ✅ "Treated firms showed improvement relative to controls."

### Misplaced Modifiers
> ❌ "We estimated the effect on firms using a DID design."
> ✅ "Using a DID design, we estimated the effect on firms."

### Anthropomorphism
> ❌ "The model wants to capture..." / "Table 1 tells us..."
> ✅ "The model captures..." / "Table 1 shows..."

### Overgeneralization
> ❌ "Trade liberalization increases growth."
> ✅ "In our sample, tariff reductions were associated with a 2.3 pp increase in GDP growth."

### Synonym Substitution (kills consistency)
> ❌ Introduction says "productivity," Methods says "output efficiency," Results says "TFP."
> ✅ Define once: "total factor productivity (TFP)" — use "TFP" consistently thereafter.

---

## Numbers and Formatting

- Spell out numbers below 10 when not attached to units ("five firms" but "5 percent")
- Never start a sentence with a numeral — spell out or restructure
- Space between number and unit: "5 mg" not "5mg"
- Use en-dash for ranges: "2010–2020" not "2010-2020"
- Thousands separator: "12,500" in English; "12 500" or "1.25万" in Chinese
- Report coefficients: match decimal precision to standard errors (if SE=0.015, report β=0.042)
- Confidence intervals: "95% CI: [0.02, 0.08]" — brackets, not parentheses

---

## Paragraph Architecture

```
[Topic sentence: one claim]
[Evidence sentence 1: data/citation supporting the claim]
[Evidence sentence 2: additional support or nuance]
[Interpretation: what this means for your argument]
[Transition: link to the next paragraph's topic]
```

- 3–7 sentences per paragraph (economics norm)
- Max 1 paragraph = 1 idea. If you have two ideas, split.
- Transitions between paragraphs: use the last sentence of Para N to set up Para N+1
- Avoid starting consecutive paragraphs with the same word

---

## Section Transitions

Good transitions create a "red thread" through the paper:

| From → To | Transition pattern |
|-----------|-------------------|
| Background → Strategy | "Given this institutional context, our identification strategy exploits..." |
| Strategy → Data | "To implement this strategy, we assemble data from..." |
| Data → Results | "Table X presents the results of estimating Equation (Y)." |
| Main results → Robustness | "A concern with our baseline is [THREAT]. We address this in Table Z." |
| Results → Conclusion | "Taken together, our findings indicate that..." |