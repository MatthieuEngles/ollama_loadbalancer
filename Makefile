# Ollama Load Balancer - Makefile
# ================================

.PHONY: help install uninstall start stop restart status logs venv deps clean test

# Variables
SERVICE_NAME = ollama-lb
SERVICE_FILE = ollama-lb.service
SYSTEMD_DIR = /etc/systemd/system
VENV_DIR = venv
PYTHON = python3

# Default target
help:
	@echo "Ollama Load Balancer - Available commands:"
	@echo ""
	@echo "  make install    - Install service (venv + deps + systemd)"
	@echo "  make uninstall  - Remove systemd service"
	@echo "  make start      - Start the service"
	@echo "  make stop       - Stop the service"
	@echo "  make restart    - Restart the service"
	@echo "  make status     - Show service status"
	@echo "  make logs       - Show service logs (follow mode)"
	@echo "  make logs-full  - Show full service logs"
	@echo ""
	@echo "  make venv       - Create virtual environment"
	@echo "  make deps       - Install Python dependencies"
	@echo "  make clean      - Remove virtual environment"
	@echo "  make run        - Run locally (without systemd)"
	@echo "  make test       - Test the API status endpoint"
	@echo ""

# Full installation
install: venv deps install-service enable-service start
	@echo ""
	@echo "========================================"
	@echo " Installation complete!"
	@echo "========================================"
	@echo ""
	@echo " Service: $(SERVICE_NAME)"
	@echo " Status:  sudo systemctl status $(SERVICE_NAME)"
	@echo " Logs:    sudo journalctl -u $(SERVICE_NAME) -f"
	@echo " API:     http://localhost:11434/api/status"
	@echo " Swagger: http://localhost:11434/docs"
	@echo ""

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment ready."

# Install dependencies
deps: venv
	@echo "Installing dependencies..."
	@$(VENV_DIR)/bin/pip install --upgrade pip -q
	@$(VENV_DIR)/bin/pip install -r requirements.txt -q
	@echo "Dependencies installed."

# Install systemd service
install-service:
	@echo "Installing systemd service..."
	@sudo cp $(SERVICE_FILE) $(SYSTEMD_DIR)/$(SERVICE_FILE)
	@sudo systemctl daemon-reload
	@echo "Service installed."

# Enable service at boot
enable-service:
	@echo "Enabling service..."
	@sudo systemctl enable $(SERVICE_NAME)
	@echo "Service enabled."

# Uninstall service
uninstall: stop
	@echo "Removing systemd service..."
	@sudo systemctl disable $(SERVICE_NAME) 2>/dev/null || true
	@sudo rm -f $(SYSTEMD_DIR)/$(SERVICE_FILE)
	@sudo systemctl daemon-reload
	@echo "Service removed."

# Start service
start:
	@echo "Starting $(SERVICE_NAME)..."
	@sudo systemctl start $(SERVICE_NAME)
	@sleep 2
	@sudo systemctl status $(SERVICE_NAME) --no-pager -l | head -20

# Stop service
stop:
	@echo "Stopping $(SERVICE_NAME)..."
	@sudo systemctl stop $(SERVICE_NAME) 2>/dev/null || true
	@echo "Service stopped."

# Restart service
restart:
	@echo "Restarting $(SERVICE_NAME)..."
	@sudo systemctl restart $(SERVICE_NAME)
	@sleep 2
	@sudo systemctl status $(SERVICE_NAME) --no-pager -l | head -20

# Reload service (after code changes)
reload: restart

# Show service status
status:
	@sudo systemctl status $(SERVICE_NAME) --no-pager -l

# Show logs (follow mode)
logs:
	@sudo journalctl -u $(SERVICE_NAME) -f

# Show full logs
logs-full:
	@sudo journalctl -u $(SERVICE_NAME) --no-pager

# Show last 100 lines of logs
logs-100:
	@sudo journalctl -u $(SERVICE_NAME) -n 100 --no-pager

# Run locally (without systemd)
run: deps
	@echo "Starting Ollama Load Balancer locally..."
	@$(VENV_DIR)/bin/python main.py

# Test API
test:
	@echo "Testing API status..."
	@curl -s http://localhost:11434/api/status | python3 -m json.tool 2>/dev/null || \
		(echo "Error: Service not responding" && exit 1)

# Test GPU detection
test-gpu:
	@echo "Testing GPU detection..."
	@$(VENV_DIR)/bin/python -c "from config import load_config; c = load_config(); print(f'GPUs: {c.gpu_ids}')"

# Clean virtual environment
clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@rm -rf __pycache__ *.pyc
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

# Update (pull latest + reinstall)
update: deps restart
	@echo "Update complete."
