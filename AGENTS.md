# Repository Guidelines

## Project Structure & Module Organization

This repository is a LaTeX thesis project built around [`main.tex`](/home/kxzhang/bachelor_final/template/template_2025/main.tex). Chapter content lives in `contents/` as separate `.tex` files. The main body of the thesis is organized as `intro`, `backend`, `system_design`, `system_impl`, `evaluation`, and `summary`, corresponding to the files in `contents/`. Supporting sections such as the abstract, acknowledgements, and appendix are stored alongside them. Bibliography data is stored in `refs.bib`. Local SJTU class files and font definitions are vendored under `texmf/` and `sjtu-text-font-newcm.def`. Figures and plotting scripts live in `figures/`, with data archives in `data/`. Project-specific agent skills are kept in `skills/`.

## Build, Test, and Development Commands

- `make`: compile `main.tex` into `main.pdf` using `latexmk`.
- `make pvc`: run continuous preview compilation while editing.
- `make wordcount`: report Chinese and total word counts via `texcount`.
- `make clean` / `make cleanall`: remove intermediate files, or intermediates plus `main.pdf`.
- `bash xelatex-local.sh main.tex`: compile with the repository-local `texmf/` tree if your TeX setup does not pick it up automatically.
- `Compile.bat thesis`: Windows entry point for the full build.

## Coding Style & Naming Conventions

Use UTF-8 and keep LaTeX source readable with one logical sentence or clause per line when practical. Prefer descriptive lowercase file names with underscores, matching the current chapter pattern in `contents/`. For Python plotting scripts in `figures/`, follow PEP 8, use 4-space indentation, and name generated assets consistently with the script or subfolder, for example `e2e_new.py` -> `e2e_new_fig.png`.

## Testing Guidelines

There is no standalone test suite. Validation is build-based:

- Run `make` before submitting changes.
- Confirm `main.pdf` compiles without LaTeX errors or missing references.
- If you modify plotting scripts or TSV inputs, rerun the script and verify the regenerated figure files under `figures/`.
- For bibliography or cross-reference edits, compile twice if needed so references settle.

## Commit & Pull Request Guidelines

Recent history favors short, imperative commit messages such as `dev: add new e2e figures`, `fix`, and `refine intro and backend`. Prefer a scoped pattern like `figs: update motivation plot` or `thesis: revise evaluation section`. Pull requests should summarize affected chapters or figures, note any regenerated assets, and attach screenshots or the updated PDF pages when layout changes are relevant.

## Repository-Specific Notes

Do not delete or rewrite vendored files under `texmf/` unless the template itself is being updated. Keep generated binaries and large archives in `data/` and generated figure outputs in their existing figure subdirectories.

## Thesis Chapter Guidance

The thesis body should follow a consistent six-chapter narrative: problem definition, background, design, implementation, evaluation, and conclusion.

- `intro`: explain the MaaS background, PD co-location challenge, research objective, significance, and the overall thesis organization. This chapter should answer why the problem matters and what this thesis contributes.
- `backend`: cover LLM inference basics, Prefill/Decode behavior, PD co-location characteristics, cluster-level scheduling, performance simulators, and prior scheduling strategies. End by identifying the concrete gap this thesis addresses.
- `system_design`: present the design goals, overall architecture, core interfaces (`Predict`, `Add_request`, `Sync_State`), state synchronization mechanism, and prediction reuse strategy. Focus on why the design choices are necessary.
- `system_impl`: describe the Rust global scheduler, vLLM instrumentation points, SSE state reporting path, predictor implementation, and online overhead optimizations. This chapter should show how the design is realized in code.
- `evaluation`: include experiment setup, baselines, hardware, workloads, end-to-end latency results, prediction accuracy, cross-table transfer analysis, and scalability or online overhead analysis. Keep the order as setup -> main results -> explanation.
- `summary`: conclude with overall contributions, key findings, current limitations, and future work. Tie the conclusions back to the original scheduling problem and the practical value of online prediction.

When editing chapter files under `contents/`, preserve this progression so each chapter naturally leads into the next one.
