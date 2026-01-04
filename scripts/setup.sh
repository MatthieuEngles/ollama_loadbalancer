#!/bin/bash
# =============================================================================
# Ollama Load Balancer - Full Setup Script
# =============================================================================
# This script sets up a fresh machine with the Ollama Load Balancer.
# It will:
#   1. Install Ollama if not present
#   2. Stop and disable the default Ollama service (if exists)
#   3. Install the load balancer as a systemd service
#   4. Wait for service to be ready
#   5. Pull models listed in startup_models.txt
#
# Usage: ./scripts/setup.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory (works even if called from another directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_FILE="$PROJECT_DIR/startup_models.txt"

echo -e "${BLUE}========================================"
echo " Ollama Load Balancer - Setup"
echo -e "========================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Install Ollama if not present
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Checking for Ollama installation...${NC}"

if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null | head -1)
    echo -e "  ${GREEN}✓ Ollama already installed: $OLLAMA_VERSION${NC}"
else
    echo "  Ollama not found, installing..."
    curl -fsSL https://ollama.com/install.sh | sh

    if command -v ollama &> /dev/null; then
        echo -e "  ${GREEN}✓ Ollama installed successfully${NC}"
    else
        echo -e "  ${RED}✗ Failed to install Ollama${NC}"
        exit 1
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Step 2: Stop and disable default Ollama service
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/5] Checking for default Ollama service...${NC}"

if systemctl list-units --full -all | grep -q "ollama.service"; then
    echo "  Found ollama.service"

    if systemctl is-active --quiet ollama; then
        echo "  Stopping ollama service..."
        sudo systemctl stop ollama
    fi

    if systemctl is-enabled --quiet ollama 2>/dev/null; then
        echo "  Disabling ollama service..."
        sudo systemctl disable ollama
    fi

    echo -e "  ${GREEN}✓ Ollama service stopped and disabled${NC}"
else
    echo -e "  ${GREEN}✓ No default ollama service found${NC}"
fi

echo ""

# -----------------------------------------------------------------------------
# Step 3: Install the load balancer
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/5] Installing Ollama Load Balancer...${NC}"

cd "$PROJECT_DIR"
make install

echo -e "${GREEN}✓ Load balancer installed${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Wait for service to be ready
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/5] Waiting for service to be ready...${NC}"

MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:11434/api/status > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Service is ready${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "  Waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "  ${RED}✗ Service did not start in time${NC}"
    echo "  Check logs with: make logs"
    exit 1
fi

echo ""

# -----------------------------------------------------------------------------
# Step 5: Pull startup models
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/5] Pulling startup models...${NC}"

if [ -f "$MODELS_FILE" ]; then
    MODEL_COUNT=0

    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$line" || "$line" == \#* ]] && continue

        MODEL_COUNT=$((MODEL_COUNT + 1))
        echo ""
        echo -e "  ${BLUE}Pulling model: $line${NC}"

        # Use the load balancer's API to pull the model
        RESPONSE=$(curl -s -X POST http://localhost:11434/api/pull \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"$line\", \"stream\": false}" \
            --max-time 600)

        if echo "$RESPONSE" | grep -q "error"; then
            echo -e "  ${RED}✗ Failed to pull $line${NC}"
            echo "  Response: $RESPONSE"
        else
            echo -e "  ${GREEN}✓ Pulled $line${NC}"
        fi

    done < "$MODELS_FILE"

    if [ $MODEL_COUNT -eq 0 ]; then
        echo "  No models found in $MODELS_FILE"
    else
        echo ""
        echo -e "  ${GREEN}✓ Pulled $MODEL_COUNT model(s)${NC}"
    fi
else
    echo "  No startup_models.txt found, skipping model pull"
    echo "  Create $MODELS_FILE with one model per line to auto-pull models"
fi

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${GREEN}========================================"
echo " Setup Complete!"
echo -e "========================================${NC}"
echo ""
echo "Service status:"
sudo systemctl status ollama-lb --no-pager -l | head -10
echo ""
echo "Useful commands:"
echo "  make status   - Check service status"
echo "  make logs     - View live logs"
echo "  make restart  - Restart service"
echo "  make test     - Test API"
echo ""
echo "API endpoints:"
echo "  http://localhost:11434/api/status   - Status"
echo "  http://localhost:11434/docs         - Swagger UI"
echo ""
