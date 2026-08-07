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
echo "🌐 Port:     ${PORT}"
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

# Start service based on environment
if [ "${SERVICE_MODE}" = "all" ] || [ "${SERVICE_MODE}" = "unified" ]; then
    echo "🚀 Starting unified dashboard + APIs on one port..."
    exec uvicorn src.unified_app:app --host 0.0.0.0 --port ${PORT} --workers 1
elif [ "${SERVICE_MODE}" = "api" ]; then
    echo "🚀 Starting advisory API (legacy standalone mode)..."
    exec uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1
elif [ "${SERVICE_MODE}" = "model-api" ]; then
    echo "🚀 Starting low-level model API (legacy standalone mode)..."
    exec uvicorn src.aerovigil_pg_bnn.api:app --host 0.0.0.0 --port ${PORT} --workers 1
elif [ "${SERVICE_MODE}" = "cli" ]; then
    echo "🔧 Running CLI inference..."
    exec aerovigil-infer "$@"
elif [ "${SERVICE_MODE}" = "gradio" ]; then
    echo "🎨 Starting Gradio demo..."
    exec python3 gradio_app/app.py
else
    echo "ℹ️  Container ready. Set SERVICE_MODE=all|api|model-api|cli|gradio."
    echo "    Recommended: docker run -e SERVICE_MODE=all -p 8000:8000 aerovigil-pg-bnn"
    exec "$@"
fi
