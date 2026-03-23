---
name: prosec-thesis-structuring
description: Structure, diagnose, and rewrite thesis outlines and chapter logic with the PROSEC principle: Problem, Related Works, Observation, Solution, Evaluation, Conclusion. Use when Codex needs to design a new thesis structure, map an existing chapter layout to PROSEC, improve chapter ordering, identify missing argumentative links, or turn scattered research notes into a coherent paper narrative for undergraduate, master's, or systems-style technical writing.
---

# Prosec Thesis Structuring

## Overview

Use this skill to turn a thesis topic, chapter list, draft, or notes into a coherent PROSEC narrative.
Favor logical progression over template compliance: every section should answer why the next section must exist.

## Core Rule

Apply PROSEC as a reasoning chain:

- `Problem`: define the practical or scientific problem, stakes, and gap.
- `Related Works`: explain what others have already done and where the gap remains.
- `Observation`: state the key empirical or conceptual insight that motivates the new approach.
- `Solution`: present the proposed method, system, model, or framework.
- `Evaluation`: verify effectiveness, limits, and tradeoffs with evidence.
- `Conclusion`: summarize findings, contributions, limitations, and future work.

Treat `Observation` as the bridge between "others have tried X" and "therefore this solution design is justified." Do not let it collapse into a vague motivation paragraph.

## Workflow

### 1. Read the user's material

Start from the artifact the user already has:

- thesis topic only
- chapter list
- section outline
- partial draft
- advisor feedback
- raw notes or experiment results

Infer the current narrative state before proposing changes.

### 2. Diagnose the current PROSEC coverage

For each PROSEC element, determine:

- whether it exists
- whether it is explicit or only implied
- whether it is in the right place
- whether it logically leads to the next element

Use a compact diagnosis table when the input is long:

| Element | Present? | Current location | Problem |
| --- | --- | --- | --- |
| Problem | yes/no | intro/ch1/etc. | too broad / missing stakes / no concrete gap |
| Related Works | yes/no | background/ch2/etc. | only summary / no critique / no gap extraction |
| Observation | yes/no | intro end/design start/etc. | missing / weak / unsupported |
| Solution | yes/no | design/impl/etc. | mechanism unclear / no link to observation |
| Evaluation | yes/no | eval/chapter x | no baseline / no explanation / weak metrics |
| Conclusion | yes/no | summary/etc. | only recap / no limitation / no future work |

### 3. Map the user's structure to PROSEC

When the user already has fixed thesis chapters, preserve the local chaptering style but realign the logic.

Use these common mappings:

- six-chapter engineering thesis:
  `intro -> Problem`, `background/backend -> Related Works`, `design -> Observation + Solution`, `implementation -> Solution realization`, `evaluation -> Evaluation`, `summary -> Conclusion`
- paper-style structure:
  `introduction -> Problem + contributions`, `related work -> Related Works`, `motivation/preliminary -> Observation`, `method -> Solution`, `experiments -> Evaluation`, `conclusion -> Conclusion`
- mixed draft with no clear structure:
  reorder sections by argumentative dependency, not by writing order

If `Observation` is too small for a standalone chapter, place it at the end of the introduction or at the beginning of the design chapter, but keep it explicit.

### 4. Produce a rewrite plan

Output one of these, depending on the request:

- a PROSEC-aligned chapter outline
- a chapter-to-PROSEC mapping
- a section rewrite plan
- a gap list with concrete fixes
- a transition plan that explains how one chapter should lead into the next

Prefer actionable phrasing such as:

- "End Chapter 2 by extracting the unresolved scheduling gap."
- "Start Chapter 3 with the observation that prediction features are reusable across scheduling states."
- "Move implementation details out of the design chapter unless they justify the mechanism."

## Writing Guidance

Apply these constraints when drafting or revising:

- Make `Problem` concrete: name the scenario, cost, or system bottleneck.
- Make `Related Works` comparative: do not list papers without extracting insufficiency.
- Make `Observation` evidence-based: derive it from measurement, system behavior, or pattern analysis.
- Make `Solution` responsive: every major mechanism should answer a specific problem or observation.
- Make `Evaluation` argumentative: report not only results but also why they support the claim.
- Make `Conclusion` bounded: summarize contributions and admit limits.

Avoid these failure modes:

- background inflation with no gap
- related work as literature summary only
- design chapter that starts before the motivating observation is clear
- evaluation with metrics but no hypothesis
- conclusion that repeats chapter summaries without synthesizing findings

## Output Pattern

When the user asks for restructuring help, prefer this response shape:

1. State the current structural diagnosis in 2-6 lines.
2. Show the target PROSEC mapping.
3. Give a revised outline or chapter responsibilities.
4. List missing links, if any.
5. Provide transition sentences or section goals when useful.

Use concise, directive language. Focus on narrative logic, not ornamental academic phrasing.

## Example Triggers

This skill should handle requests like:

- "Use PROSEC to reorganize my thesis outline."
- "Map my six thesis chapters to Problem, Related Works, Observation, Solution, Evaluation, Conclusion."
- "My draft has background and design, but the logic feels disconnected. Diagnose it with PROSEC."
- "Turn these research notes into a thesis structure."
- "Where should Observation appear in a Chinese undergraduate engineering thesis?"

## Self-Check

Before finalizing a PROSEC-based rewrite, verify:

- the problem statement leads to a concrete unresolved gap
- related work ends with a limitation that motivates the thesis
- observation is explicit and design-relevant
- the solution answers the diagnosed gap instead of introducing unrelated mechanisms
- evaluation tests the main claim with appropriate baselines or comparisons
- the conclusion reflects actual findings rather than generic claims
