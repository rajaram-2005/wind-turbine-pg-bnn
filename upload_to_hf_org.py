from huggingface_hub import HfApi, login
from pathlib import Path

# ─── CONFIGURATION ─────────────────────────────────────────────
ORG_NAME = "AerovigilAI"                        # Organization namespace
MODEL_NAME = "wind-turbine-pg-bnn"
REPO_ID = f"{ORG_NAME}/{MODEL_NAME}"            # Full repo ID

MODEL_PATH = "artifacts/bnn_demo.pt"            # Local model weights
README_PATH = "README.md"                       # Model card
CONFIG_PATH = "config.json"                     # Model configuration
HF_TOKEN = None  # Set to your token string, or use `huggingface-cli login`
# ───────────────────────────────────────────────────────────────

# Authenticate (uses cached token if HF_TOKEN is None)
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Using cached Hugging Face token. Run `huggingface-cli login` if needed.")

# Initialize API
api = HfApi()

# Create repo under Aerovigil AI organization
# Note: You must be a member/admin of the AerovigilAI organization
api.create_repo(
    repo_id=REPO_ID,
    exist_ok=True,
    private=False,           # Public repo
    repo_type="model"
)

# Upload all files
print(f"📤 Uploading to https://huggingface.co/{REPO_ID} ...")

api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="bnn_demo.pt",
    repo_id=REPO_ID,
)

api.upload_file(
    path_or_fileobj=README_PATH,
    path_in_repo="README.md",
    repo_id=REPO_ID,
)

api.upload_file(
    path_or_fileobj=CONFIG_PATH,
    path_in_repo="config.json",
    repo_id=REPO_ID,
)

print(f"✅ Successfully uploaded to https://huggingface.co/{REPO_ID}")
print(f"\n🔗 View your model: https://huggingface.co/{REPO_ID}")
print(f"📝 Update model card: https://huggingface.co/{REPO_ID}/edit/main/README.md")
