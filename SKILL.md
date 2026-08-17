---
name: solve-math-modeling
description: Solve mathematical-modeling problems end to end, from problem decomposition and model selection through data auditing, mathematical formulation, executable code, visualization, validation, sensitivity analysis, and competition-paper delivery. Use for MCM/ICM, China Undergraduate Mathematical Contest in Modeling and similar optimization, prediction, evaluation, simulation, mechanism, network, spatial, or mixed modeling tasks; also use when improving an existing modeling solution, checking its reproducibility, or turning partial analysis and data files into a rigorous report.
---

# Solve Mathematical Modeling Problems

## Operating contract

Produce an evidence-backed, reproducible solution rather than a catalogue of models. Keep every formula, code result, figure, and conclusion traceable to the problem statement, supplied data, an explicitly cited external source, or a clearly labeled assumption.

Never:

- invent empirical data, solver output, accuracy, rankings, or citations;
- describe unexecuted code as successfully run;
- use test data during training or preprocessing fitting;
- force a fashionable model where a simpler model is adequate;
- hide infeasibility, unstable estimates, failed diagnostics, or material limitations.

If required data are absent, distinguish mandatory from optional data. Continue with symbolic derivation or explicitly labeled scenario/synthetic data only when useful, and state what cannot yet be concluded.

## Start by establishing the task contract

1. Inspect the full prompt, attachments, tables, units, date ranges, and required deliverables.
2. Infer the user's language, programming proficiency, preferred stack, contest target, and emphasis. Ask only for information that would materially change the solution and cannot be inferred.
3. Create a requirement ledger containing:
   - each subproblem and its direct and hidden objective;
   - known inputs, unknowns, constraints, outputs, and evaluation criteria;
   - dependencies among subproblems;
   - required artifacts: formulas, code, figures, tables, paper, or files.
4. State the current evidence boundary: supplied data, allowed external data, and assumptions.

Read [problem-and-model-routing.md](references/problem-and-model-routing.md) when decomposing the question or choosing models.

## Execute the seven-stage workflow

### 1. Analyze the problem

- Rewrite each subproblem in operational terms: given data -> decision or inference -> constraints -> output.
- Classify it as optimization, prediction, evaluation, mechanism/inference, simulation, network/spatial, or a justified combination.
- Draw a dependency map when there are at least three linked subproblems.
- Identify ambiguities and resolve them by text evidence, domain convention, or explicit assumption in that order.

### 2. Audit and prepare data

- Preserve raw files; work on copies and maintain a data dictionary with source, unit, range, and meaning.
- Profile schema, missingness, duplicates, impossible values, outliers, imbalance, temporal order, and sample size before transforming data.
- Run `scripts/audit_dataset.py` for CSV/TSV/XLSX files when practical.
- Choose missing-value, outlier, scaling, encoding, aggregation, and feature construction methods based on the data-generating process and downstream model.
- Fit preprocessing only on training data. Use time-ordered or grouped splits where random splitting would leak information.
- Record every deletion, imputation, correction, transformation, and external-data merge.

Read [data-and-validation.md](references/data-and-validation.md) for preprocessing, split strategy, metrics, diagnostics, and uncertainty.

### 3. Select models

- Propose at most two serious candidates per subproblem unless the user requests a broader benchmark.
- Prefer one interpretable baseline and one justified improvement when comparison adds value; do not manufacture novelty.
- Compare candidates on assumptions, fit to the question, data needs, interpretability, computational cost, validation plan, and failure modes.
- Select a primary model using evidence. Keep alternatives as benchmarks or contingency plans.
- For every chosen model, explain its principle in plain language, suitability, meaningful improvement, and limitations.

### 4. Formulate the model

- Define decision, state, intermediate, and target variables in a table with symbol, type, unit, domain, and source.
- List 3-5 core assumptions, each with rationale, consequence, and a possible validation or relaxation.
- Derive the objective, constraints, transition/observation equations, and evaluation quantities step by step.
- Explain each formula immediately after it. Check dimensions, domains, boundary cases, identifiability, feasibility, and direction of inequalities.
- Make the chain `variables -> parameters -> equations -> constraints -> objective/output -> validation` visible.

Read [formulation-and-solving.md](references/formulation-and-solving.md) before writing a full formulation or implementation.

### 5. Solve, code, and visualize

- Implement the model in the user's requested language; default to Python when none is specified.
- Provide installation commands and a single reproducible entry point. Set seeds where randomness is used.
- Organize code as input -> validation -> preprocessing -> model -> evaluation -> export.
- Add comments that explain modeling decisions, units, bounds, and non-obvious numerical choices.
- Run code when an execution environment and data are available. Report the command, dependency versions where relevant, outputs, warnings, and failures.
- Select figures by analytical purpose: distributions, relationships, trends, residuals, spatial patterns, optimization trade-offs, or sensitivity. Include title, axes with units, legend, readable scale, and a one-sentence finding.
- Save machine-readable results separately from presentation figures.

### 6. Validate and analyze results

- Establish a baseline or naive comparator.
- Use validation appropriate to the task: holdout/cross-validation/backtesting, residual diagnostics, constraint and feasibility checks, convergence checks, calibration, conservation laws, or comparison with known cases.
- Report both central performance and uncertainty or variability. Use multiple metrics when one metric can conceal important errors.
- Perform sensitivity analysis on influential parameters and robustness checks on assumptions, preprocessing, model specification, and data perturbations.
- Separate numerical description from interpretation. Link every major conclusion back to the question and constraints.
- If a result is implausible, investigate units, leakage, indexing, signs, bounds, overfitting, and solver status before interpreting it.

### 7. Write and package the deliverables

- Start from [competition-report-template.md](assets/competition-report-template.md) when a paper or full solution is required.
- Keep symbols, units, terminology, sample counts, tables, and figures consistent across text, formulas, code, and appendices.
- Present each subproblem as: objective -> method -> formulation -> solution -> validation -> conclusion.
- State strengths, limitations, scope of applicability, and actionable recommendations without exaggeration.
- Include data provenance, software/dependencies, random seeds, run command, and output paths.
- Perform the final audit below before delivery.

Read [results-and-writing.md](references/results-and-writing.md) when interpreting outputs or drafting the report.

## Final audit

Do not declare completion until all applicable checks pass:

- **Coverage:** every subproblem and requested artifact is addressed.
- **Traceability:** each conclusion maps to data, a formula, code output, or an explicit assumption.
- **Mathematics:** symbols are defined; units, domains, bounds, and constraint directions are consistent.
- **Data:** raw data are preserved; transformations and exclusions are documented; leakage is prevented.
- **Code:** the provided command runs or failures are disclosed; paths are portable; seeds and dependencies are recorded.
- **Validation:** baselines, diagnostics, uncertainty, sensitivity, and robustness are proportionate to the claims.
- **Visuals:** figures answer specific questions and match the reported numbers.
- **Writing:** no unsupported superlatives, fake precision, fabricated citations, or conclusions beyond the evidence.

## Default response structure

For a complete problem, deliver:

1. Problem decomposition and dependency map
2. Data inventory and audit
3. Candidate-model comparison and final selection
4. Assumptions and notation table
5. Model derivation by subproblem
6. Reproducible code and run instructions
7. Results, figures, diagnostics, sensitivity, and robustness
8. Conclusions, limitations, and recommendations
9. Paper-ready outline or requested files

For a partial request, execute only the relevant stages but preserve the same evidence and validation standards.
