"""
Aerovigil AI: Wind Turbine RUL Prediction Demo
Deployed on Hugging Face Spaces: https://huggingface.co/spaces/AerovigilAI/wind-turbine-rul-demo
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from huggingface_hub import hf_hub_download

# ─── MODEL LOADING ─────────────────────────────────────────────
@gr.load
def load_model():
    """Load PG-BNN model from Hugging Face Hub."""
    repo_id = "AerovigilAI/wind-turbine-pg-bnn"

    # Download files
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    model_path = hf_hub_download(repo_id=repo_id, filename="bnn_demo.pt")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Import model class (from installed package or local)
    try:
        from aerovigil_pg_bnn import PhysicsGuidedBNN
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from aerovigil_pg_bnn import PhysicsGuidedBNN

    # Load model
    model = PhysicsGuidedBNN(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    return model, config


# ─── PREDICTION FUNCTION ───────────────────────────────────────
def predict_rul(vibration_rms, bearing_temp, generator_temp, power_output,
                wind_speed, operating_hours, n_samples):
    """Run MCVI inference and return predictions."""

    # Load model and config
    model, config = load_model()

    # Prepare input tensor
    input_data = torch.tensor([[
        vibration_rms,
        bearing_temp,
        generator_temp,
        power_output,
        wind_speed,
        operating_hours,
    ]], dtype=torch.float32)

    # Run MCVI inference
    model.train()  # Enable dropout
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            rul_mean, _ = model(input_data)
            predictions.append(rul_mean.item())

    # Compute statistics
    predictions = np.array(predictions)
    mean_rul = float(np.mean(predictions))
    std_rul = float(np.std(predictions))
    ci_lower = float(np.percentile(predictions, 2.5))
    ci_upper = float(np.percentile(predictions, 97.5))

    # Risk assessment
    if mean_rul < 14:
        risk_level = "🔴 CRITICAL"
        recommendation = "🚨 URGENT: Immediate maintenance required! Risk of failure within 2 weeks. Contact maintenance team immediately."
    elif mean_rul < 30:
        risk_level = "🟠 HIGH"
        recommendation = "⚠️ Schedule maintenance within 2-4 weeks. Increased monitoring recommended. Prepare spare parts."
    elif mean_rul < 45:
        risk_level = "🟡 MODERATE"
        recommendation = "📅 Schedule routine maintenance within 45 days. Continue monitoring trends."
    else:
        risk_level = "🟢 LOW"
        recommendation = "✅ Normal operation. Continue standard monitoring. Next inspection in 30 days."

    # Distribution plot
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=predictions,
        nbinsx=30,
        name="RUL Distribution",
        marker_color="#3b82f6",
        opacity=0.7,
    ))

    # Add vertical lines
    fig.add_vline(x=mean_rul, line_dash="solid", line_color="#1f2937",
                  annotation_text=f"Mean: {mean_rul:.1f} days")
    fig.add_vline(x=ci_lower, line_dash="dash", line_color="#6b7280",
                  annotation_text=f"95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")
    fig.add_vline(x=ci_upper, line_dash="dash", line_color="#6b7280")

    # Threshold lines
    fig.add_vline(x=45, line_dash="dot", line_color="#ea580c",
                  annotation_text="45-day threshold")
    fig.add_vline(x=14, line_dash="dot", line_color="#dc2626",
                  annotation_text="Critical threshold")

    fig.update_layout(
        title="Predicted Remaining Useful Life Distribution",
        xaxis_title="RUL (days)",
        yaxis_title="Frequency",
        showlegend=False,
        template="plotly_white",
        height=400,
    )

    return {
        "mean_rul": round(mean_rul, 1),
        "uncertainty": round(std_rul, 1),
        "ci_95": [round(ci_lower, 1), round(ci_upper, 1)],
        "risk_level": risk_level,
        "recommendation": recommendation,
        "distribution_plot": fig,
        "maintenance_urgent": mean_rul < 14,
    }


# ─── GRADIO INTERFACE ──────────────────────────────────────────
def create_interface():
    with gr.Blocks(
        title="Aerovigil AI — Wind Turbine RUL Predictor",
        theme=gr.themes.Soft(),
        css="""
        .risk-critical { background: #fee2e2; border-left: 4px solid #dc2626; padding: 1rem; }
        .risk-high { background: #ffedd5; border-left: 4px solid #ea580c; padding: 1rem; }
        .risk-moderate { background: #fef9c3; border-left: 4px solid #ca8a04; padding: 1rem; }
        .risk-low { background: #dcfce7; border-left: 4px solid #16a34a; padding: 1rem; }
        """
    ) as demo:

        # Header
        gr.Markdown("""
        # ⚡ Aerovigil AI: Wind Turbine RUL Predictor

        **Physics-Guided Bayesian Neural Network** for drivetrain bearing
        Remaining Useful Life (RUL) prediction with uncertainty quantification.

        | Metric | Value |
        |--------|-------|
        | Early Warning Horizon | **45 days** |
        | Accuracy | **94.2%** |
        | Recall | **100%** |

        [🌐 Live App](https://aerovigil.abacusai.app) |
        [📦 Model](https://huggingface.co/AerovigilAI/wind-turbine-pg-bnn) |
        [💻 GitHub](https://github.com/rajaram-2005/wind-turbine-pg-bnn)
        """)

        with gr.Row():
            # Input Panel
            with gr.Column(scale=1):
                gr.Markdown("## 📊 SCADA Telemetry Input")

                vibration_rms = gr.Slider(
                    minimum=0, maximum=50, value=12.5,
                    label="Vibration RMS (mm/s)",
                    info="Drive-train vibration severity"
                )
                bearing_temp = gr.Slider(
                    minimum=20, maximum=150, value=65.0,
                    label="Bearing Temperature (°C)",
                    info="Main bearing operating temperature"
                )
                generator_temp = gr.Slider(
                    minimum=20, maximum=200, value=80.0,
                    label="Generator Temperature (°C)",
                    info="Generator winding temperature"
                )
                power_output = gr.Slider(
                    minimum=0, maximum=5000, value=2100.0,
                    label="Power Output (kW)",
                    info="Active power generation"
                )
                wind_speed = gr.Slider(
                    minimum=0, maximum=30, value=12.0,
                    label="Wind Speed (m/s)",
                    info="Nacelle anemometer reading"
                )
                operating_hours = gr.Slider(
                    minimum=0, maximum=100000, value=35000.0,
                    label="Operating Hours",
                    info="Cumulative turbine runtime"
                )

                n_samples = gr.Slider(
                    minimum=10, maximum=500, value=100, step=10,
                    label="MCVI Samples",
                    info="More samples = smoother uncertainty"
                )

                predict_btn = gr.Button("🔮 Predict RUL", variant="primary")

                gr.Markdown("""
                ---
                ### 📝 Input Guidelines
                - **Vibration RMS**: Normal < 10, Alert 10-18, Danger > 18 mm/s
                - **Bearing Temp**: Normal < 80°C, Alert 80-100°C, Danger > 100°C
                - Typical turbine operates 15-25 m/s wind speed
                """)

            # Output Panel
            with gr.Column(scale=1):
                gr.Markdown("## 📈 Prediction Results")

                with gr.Group():
                    risk_badge = gr.HTML(label="Risk Assessment")

                    with gr.Row():
                        mean_rul = gr.Number(label="Mean RUL (days)", interactive=False)
                        uncertainty = gr.Number(label="Uncertainty (±days)", interactive=False)

                    ci_display = gr.JSON(label="95% Confidence Interval", interactive=False)
                    recommendation = gr.Textbox(
                        label="Maintenance Recommendation",
                        lines=3,
                        interactive=False
                    )

                    plot = gr.Plot(label="RUL Distribution")

        # Event handlers
        def update_ui(result):
            risk_class = {
                "🔴 CRITICAL": "risk-critical",
                "🟠 HIGH": "risk-high",
                "🟡 MODERATE": "risk-moderate",
                "🟢 LOW": "risk-low",
            }.get(result["risk_level"], "risk-low")

            html = f"""
            <div class="{risk_class}">
                <h3>{result['risk_level']}</h3>
                <p><strong>Mean RUL:</strong> {result['mean_rul']} days</p>
                <p><strong>Uncertainty:</strong> ±{result['uncertainty']} days</p>
            </div>
            """

            return (
                html,
                result["mean_rul"],
                result["uncertainty"],
                result["ci_95"],
                result["recommendation"],
                result["distribution_plot"],
            )

        predict_btn.click(
            fn=predict_rul,
            inputs=[vibration_rms, bearing_temp, generator_temp,
                    power_output, wind_speed, operating_hours, n_samples],
            outputs=gr.State(),
        ).then(
            fn=update_ui,
            inputs=gr.State(),
            outputs=[risk_badge, mean_rul, uncertainty, ci_display, recommendation, plot],
        )

        # Example presets
        gr.Examples(
            examples=[
                [8.2, 55.0, 72.0, 1800.0, 11.5, 12000.0, 100],   # Healthy
                [15.7, 78.0, 89.0, 2100.0, 13.2, 35000.0, 100],  # Moderate wear
                [28.3, 95.0, 110.0, 1950.0, 12.8, 52000.0, 100], # High wear
                [42.1, 118.0, 135.0, 1600.0, 10.5, 68000.0, 100],# Critical
            ],
            inputs=[vibration_rms, bearing_temp, generator_temp,
                    power_output, wind_speed, operating_hours, n_samples],
            label="⚡ Example Scenarios",
        )

        # Footer
        gr.Markdown("""
        ---
        ### 🧠 About This Model

        This **Physics-Guided Bayesian Neural Network** integrates:
        - **ISO 281 bearing fatigue life theory** as physics constraints
        - **Monte Carlo Variational Inference** for uncertainty quantification
        - **SCADA telemetry fusion** from operational wind farms

        The 45-day early warning horizon enables proactive maintenance scheduling,
        reducing unplanned downtime by up to 60%.

        **Cite this model:**
        ```bibtex
        @software{aerovigil_pgbnn_2024,
          author = {Aerovigil AI},
          title = {Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction},
          year = {2024},
          url = {https://huggingface.co/AerovigilAI/wind-turbine-pg-bnn}
        }
        ```
        """)

    return demo


# Launch
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
