# config
PYTHON = python3
SRC_DIR = src
DATA = data.csv
THETA_FILE = theta.txt
PLOT_FILE = regression_plot.png

# venv
VENV_DIR = .venv
PYTHON_VENV = $(VENV_DIR)/bin/python
PIP_VENV = $(VENV_DIR)/bin/pip

# colours
GREEN = \033[0;32m
BLUE = \033[0;34m
YELLOW = \033[0;33m
RED = \033[0;31m
BOLD = \033[1m
NC = \033[0m

.PHONY: all venv train predict visual run test clean venv-clean fclean re check-model

# default
all: run

venv:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "$(YELLOW)Virtual environment already exists.$(NC)"; \
		echo "$(YELLOW)Run 'make venv-clean' first to recreate.$(NC)"; \
	else \
		echo "$(BLUE)Creating virtual environment...$(NC)"; \
		python3 -m venv $(VENV_DIR); \
		echo "$(BLUE)Installing dependencies...$(NC)"; \
		$(PIP_VENV) install --upgrade pip; \
		$(PIP_VENV) install matplotlib; \
		echo "$(GREEN)✓ Virtual environment ready!$(NC)"; \
		echo ""; \
		echo "$(YELLOW)$(BOLD)Next steps:$(NC)"; \
		echo "  1. $(GREEN)source $(VENV_DIR)/bin/activate$(NC)"; \
		echo "  2. $(GREEN)make$(NC)"; \
	fi

train:
	@echo "$(BLUE)$(BOLD)Training the model...$(NC)"
	@$(PYTHON) $(SRC_DIR)/learn.py
	@echo "$(GREEN)✓ Training complete!$(NC)\n"

predict: check-model
	@echo "$(BLUE)$(BOLD)Running prediction...$(NC)"
	@$(PYTHON) $(SRC_DIR)/predict.py

check-model:
	@if [ ! -f $(THETA_FILE) ]; then \
		echo "$(YELLOW)⚠ Model not found. Training first...$(NC)\n"; \
		$(MAKE) train; \
		echo ""; \
	fi

visual: check-model
	@echo "$(BLUE)$(BOLD)Generating visualization...$(NC)"
	@$(PYTHON) $(SRC_DIR)/visual.py
	@echo "$(GREEN)✓ Plot saved as $(PLOT_FILE)$(NC)\n"

run: train visual
	@echo "$(GREEN)$(BOLD)✓ Complete workflow finished!$(NC)"

workflow: train
	@echo ""
	@echo "$(BLUE)$(BOLD)Testing predictions with sample values...$(NC)"
	@echo "$(YELLOW)─────────────────────────────────────────$(NC)"
	@echo ""
	@echo "$(GREEN)Testing 50,000 km:$(NC)"
	@echo "50000" | $(PYTHON) $(SRC_DIR)/predict.py
	@echo ""
	@echo "$(GREEN)Testing 100,000 km:$(NC)"
	@echo "100000" | $(PYTHON) $(SRC_DIR)/predict.py
	@echo ""
	@echo "$(GREEN)Testing 150,000 km:$(NC)"
	@echo "150000" | $(PYTHON) $(SRC_DIR)/predict.py
	@echo ""
	@echo "$(GREEN)$(BOLD)✓ Testing complete!$(NC)"

test: workflow visual
	@echo ""
	@echo "$(GREEN)$(BOLD)✓ All tests complete!$(NC)"

clean:
	@echo "$(RED)Cleaning generated files...$(NC)"
	@rm -f $(THETA_FILE)
	@rm -f $(PLOT_FILE)
	@echo "$(GREEN)✓ Cleaned $(THETA_FILE) and $(PLOT_FILE)$(NC)"

venv-clean:
	@echo "$(RED)Removing virtual environment...$(NC)"
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"

fclean: clean venv-clean
	@echo "$(RED)Cleaning Python cache...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned all cache files$(NC)"

re: fclean train
	@echo "$(GREEN)$(BOLD)✓ Retrained from scratch!$(NC)"
