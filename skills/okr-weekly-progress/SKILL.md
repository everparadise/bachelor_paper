---
name: okr-weekly-progress
description: Draft weekly progress reports by mapping completed work to the user's OKR and grouping work into clear categories. Use when Codex needs to turn raw weekly task notes, daily logs, chat fragments, or bullet lists into an OKR-aligned weekly update, progress fill-in, work classification, or advisor/mentor status report, including requests such as "根据我这周做的事情，对照我的OKR撰写周报和工作分类" and "对照我的OKR填写工作进展，用于向导师汇报".
---

# OKR Weekly Progress

## Overview

Turn scattered weekly work items into a concise, evidence-based progress update aligned to OKR.
Extract objectives first, map every task to the most relevant objective or key result, then write the report in the user's language and tone.

## Workflow

1. Collect the minimum inputs.
2. Normalize weekly work items into atomic tasks.
3. Map each task to OKR.
4. Summarize progress, outcomes, and next steps.
5. Produce a report that is easy to paste into a weekly update or advisor message.

## Collect Inputs

Request or extract:

- The user's OKR text.
- The raw list of work completed this week.
- Optional constraints: audience, tone, word limit, report format, language.

If OKR or weekly work is missing, ask only for the missing input. Do not invent OKR content.

## Normalize Weekly Work

Rewrite raw notes into atomic work items.
Each item should capture:

- What was done.
- Which artifact or result was produced.
- Whether the task is completed, ongoing, blocked, or deferred.
- Any measurable evidence already present in the input.

Merge duplicates and separate bundled statements like "调研+实现+测试" into distinct steps when the distinction matters for OKR mapping.

## Map Work to OKR

For each work item:

- Match it to the most relevant objective or key result.
- State the mapping explicitly.
- If the alignment is weak, mark it as indirect support instead of forcing a direct match.
- If a task does not fit any OKR, place it in "other/supporting work" and say why.

Prefer evidence over enthusiasm. Do not overclaim impact that is not supported by the provided work log.

## Write the Report

Default output should include:

- A brief weekly summary.
- Work categories grouped by OKR or objective.
- A progress section describing what advanced this week.
- Risks or blockers when present.
- Next-step suggestions for the coming week.

Use the user's language by default. If the prompt is Chinese, answer in Chinese unless asked otherwise.
When the audience is a mentor, advisor, or supervisor, keep the tone factual and compact.

## Classification Rules

Use one of these grouping modes based on the request:

- By objective: best when the user wants direct OKR alignment.
- By work type: research, implementation, testing, documentation, communication, project coordination.
- Hybrid: primary grouping by objective, secondary labels by work type.

If the user says "工作分类", prefer the hybrid mode unless they specify another structure.

## Output Requirements

Do not just paraphrase the task list.
Convert activities into progress statements with outcomes, for example:

- Bad: "阅读了论文，修改了代码，做了测试。"
- Better: "围绕 KR2 完成论文调研与实现验证，补齐关键代码修改并完成基础测试，为后续实验对比提供了可复用结果。"

Keep uncertainty explicit:

- Use "已完成", "进行中", "待验证", "存在阻塞" when status is known.
- Use "根据现有信息推断" when inferring a category or impact.

## Resource Use

Load [weekly-report-template.md](./references/weekly-report-template.md) when the user needs a ready-to-paste structure, an advisor-facing format, or a compact Chinese template.
