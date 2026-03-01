# ── octopus-allocation Makefile ───────────────────────────────────────────────
PYTHON  ?= python3
VENV    := venv
PIP     := $(VENV)/bin/pip
PYRUN   := $(VENV)/bin/python3

TEX_DIR := doc/v1
TEX     := $(TEX_DIR)/v1.tex
LATEXMK := latexmk
PDFLATEX := $(HOME)/.TinyTeX/bin/x86_64-linux/pdflatex
BIBTEX   := $(HOME)/.TinyTeX/bin/x86_64-linux/bibtex
export PATH := $(HOME)/.TinyTeX/bin/x86_64-linux:$(PATH)

# ── venv setup ────────────────────────────────────────────────────────────────
.PHONY: venv
venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q
	@echo "✓  venv ready — activate with: source venv/bin/activate"

# ── plots ─────────────────────────────────────────────────────────────────────
.PHONY: plots
plots: venv
	$(PYRUN) plot_demand.py

# ── LaTeX pdf ─────────────────────────────────────────────────────────────────
.PHONY: pdf
pdf: plots
	cd $(TEX_DIR) && \
	  pdflatex -interaction=nonstopmode v1.tex && \
	  bibtex v1 && \
	  pdflatex -interaction=nonstopmode v1.tex && \
	  pdflatex -interaction=nonstopmode v1.tex
	@echo "✓  PDF written to $(TEX_DIR)/v1.pdf"

# ── quick recompile (no figure regen) ─────────────────────────────────────────
.PHONY: tex
tex:
	cd $(TEX_DIR) && \
	  pdflatex -interaction=nonstopmode v1.tex
	@echo "✓  PDF written to $(TEX_DIR)/v1.pdf"

# ── clean LaTeX build artefacts ───────────────────────────────────────────────
.PHONY: clean
clean:
	rm -f $(TEX_DIR)/*.aux $(TEX_DIR)/*.log $(TEX_DIR)/*.out \
	       $(TEX_DIR)/*.toc $(TEX_DIR)/*.bbl $(TEX_DIR)/*.blg \
	       $(TEX_DIR)/*.fls $(TEX_DIR)/*.fdb_latexmk \
	       $(TEX_DIR)/*.synctex.gz

.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV)
	rm -rf $(TEX_DIR)/figs
