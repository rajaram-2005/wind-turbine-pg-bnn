#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Aerovigil AI — Wind Turbine RUL Prediction Service   ║"
echo "║     Physics-Guided Bayesian Neural Network               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📦 Model:    ${MODEL_PATH}"
echo "⚙️  Config:   ${CONFIG_PATH}"
if [ -n "${SCALER_PATH}" ] && [ -f "${SCALER_PATH}" ]; then
    echo "📊 Scaler:   ${SCALER_PATH}"
fi
echo "🌐 Port:     ${PORT:-8080}"
echo ""

# Verify model file exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "❌ Error: Model file not found at ${MODEL_PATH}"
    exit 1
fi

# Verify config exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "❌ Error: Config file not found at ${CONFIG_PATH}"
    exit 1
fi

# Check PyTorch and CUDA availability
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

# The production image has one network boundary. Historical service-mode names
# are accepted for compatibility, but they all converge on the unified app
# instead of creating a second API or Gradio process.
case "${SERVICE_MODE:-all}" in
    all|unified)
        echo "🚀 Starting unified console + APIs on one port..."
        exec uvicorn src.unified_app:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
        ;;
    api|model-api|gradio)
        echo "⚠️  SERVICE_MODE=${SERVICE_MODE} is deprecated; starting the unified app instead."
        exec uvicorn src.unified_app:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
        ;;
    cli)
        echo "🔧 Running CLI inference..."
        exec aerovigil-infer "$@"
        ;;
    *)
        if [ "$#" -eq 0 ]; then
            echo "❌ Unknown SERVICE_MODE=${SERVICE_MODE}. Use all|unified|cli or provide a command."
            exit 2
        fi
        exec "$@"
        ;;
esac
