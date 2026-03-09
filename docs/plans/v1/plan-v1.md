# Cursor Prompt: Build a LaTeX Report for RL Systems Project

## Your Task
You are helping me build a LaTeX document (`.tex`) for a technical report/presentation on an RL-based control system project. Follow the plan below exactly. Before writing any LaTeX, you must:

1. **Read this document fully**
2. **Ask me clarifying questions** (listed at the bottom) one section at a time
3. **Propose a plan** (document class, structure, packages) and wait for my approval before proceeding
4. **Only begin writing LaTeX after I explicitly say "go ahead" or equivalent**

---

## Document Structure

The final `.tex` file should follow this outline:

```
1. RL Systems Work
   1.1 Problem Formulation
       - Source: borrow from my slides and docs (ASK ME for files)
   1.2 Why is RL a Good Solution?
       - Begin with a short intro to RL
       1.2.1 Data Plotting
           - Agent uses traces to plot data
           - Show how this is done using the .ipynb (ASK ME for the notebook)
       1.2.2 Assuming RL is NOT the solution (baseline comparisons)
           - PID-Based Baseline
               * Short intro to PID control
           - Greedy + Optimal Baseline
               * Source: pull from .ipynb, slides, and docs
   1.3 Building the RL Policy and Environment
       - Source: borrow from slides
       1.3.1 States
       1.3.2 Actions
       1.3.3 Reward
       1.3.4 Model — short intro on each:
           - LCPO (Lyapunov-based Constrained Policy Optimization)
           - SAC (Soft Actor-Critic)
   1.4 Next Steps: How to Verify the Results
```

---

## Cursor's Proposed Plan (Fill This Out Before Starting)

Before writing any LaTeX, propose answers to the following and wait for my approval:

### Document Class & Style
- What document class should be used? (`article`, `report`, `beamer` for slides, `IEEEtran`, custom)?
- Should this be a **report** (long-form prose) or a **presentation** (beamer slides)?
- Any specific journal/conference template required?

### Packages to Use
Propose a package list. Likely candidates:
- `amsmath`, `amssymb` — math
- `graphicx` — figures
- `hyperref` — links
- `listings` or `minted` — code blocks (for ipynb content)
- `booktabs` — tables
- `geometry` — margins
- `biblatex` or `natbib` — references
- `algorithm2e` or `algorithmicx` — pseudocode (for PID, RL algorithms)

### Content Sourcing Questions (Ask Me These)
Before writing each section, ask:

1. **Problem Formulation**: Can you share the slides/docs? Should I extract text directly or paraphrase into LaTeX prose?
2. **RL Intro**: How long should this be — a paragraph, half a page, full page?
3. **Data Plotting**: Should the `.ipynb` code be included as a `listings` block, or just described in prose with figures?
4. **PID Baseline**: Is there existing text/equations to pull from, or should I write a generic PID intro?
5. **Greedy + Optimal**: What specifically from the `.ipynb` and slides should be included here?
6. **States/Actions/Reward**: Should each be a subsection with prose, a table, or a bullet list?
7. **LCPO & SAC intros**: How short is "short"? (~1 paragraph each? with equations?)
8. **Next Steps**: Is this a to-do list, a proposed experimental protocol, or prose discussion?
9. **Figures**: Are there plots/diagrams to include? If so, what format (`.pdf`, `.png`)?
10. **References**: Should citations be included? BibTeX file available?

---

## Workflow Instructions for Cursor

Follow this loop:

```
LOOP:
  1. Ask me the next batch of clarifying questions (group related ones together)
  2. Wait for my response
  3. Summarize what you understood and confirm before proceeding
  4. Write only the section(s) I've approved
  5. Show me the output and ask if I want changes before moving on
END LOOP when all sections are complete.
```

Do **not** write placeholder text like `TODO` or `[INSERT CONTENT]` unless I explicitly say that's okay. If you don't have the source material for a section, ask me for it instead of guessing.

---

## Final Output
- A single `.tex` file (or split into `\input{}` files if I prefer — ask me)
- A `references.bib` if citations are needed
- Compilable with `pdflatex` or `xelatex` (ask which I prefer)