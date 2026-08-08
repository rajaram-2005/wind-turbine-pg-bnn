const form = document.querySelector('#advisory-form');
const submit = document.querySelector('#submit');
const value = (form, key) => form.elements[key].value;
const number = (key) => Number(value(form, key));

function optionalNumber(key) {
  const raw = value(form, key).trim();
  return raw === '' ? undefined : Number(raw);
}
function showError(message) {
  const error = document.querySelector('#error');
  error.textContent = message;
  error.hidden = false;
}
function render(data) {
  document.querySelector('#empty').hidden = true;
  document.querySelector('#error').hidden = true;
  document.querySelector('#result').hidden = false;
  document.querySelector('#result-asset').textContent = data.asset_id;
  document.querySelector('#rul').textContent = `${data.predicted_rul_days.toFixed(1)} days`;
  document.querySelector('#inspection').textContent = `${data.suggested_inspection_window_days.toFixed(1)} days`;
  document.querySelector('#uncertainty').textContent = data.epistemic_std.toFixed(3);
  const team = data.agent_team || {};
  const risk = document.querySelector('#risk');
  risk.textContent = team.risk_level || 'REVIEW';
  risk.className = `risk ${(team.risk_level || '').toLowerCase()}`;
  document.querySelector('#summary').textContent = team.shared_summary || 'Connected model and safety evidence available.';
  document.querySelector('#sources').innerHTML = (team.connected_sources || []).map(s => `<li>${s.replaceAll('_', ' ')}</li>`).join('');
  document.querySelector('#rationale').textContent = data.rationale;
  const physics = data.physics_guided;
  document.querySelector('#physics-section').hidden = !physics;
  if (physics) {
    document.querySelector('#physics').textContent = `${physics.target_name}: ${physics.target_mean.toFixed(3)} ± ${physics.total_std.toFixed(3)}`;
    document.querySelector('#provenance').textContent = `Feature provenance: ${Object.entries(physics.feature_sources).map(([k,v]) => `${k} (${v})`).join(' · ')}`;
  }
}
async function importHardware(payload) {
  const status = document.querySelector('#hardware-status');
  status.textContent = 'Validating hardware telemetry…';
  try {
    const response = await fetch('/api/hardware/ingest', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    for (const [key, value] of Object.entries(data.latest_telemetry)) form.elements[key].value = value;
    status.textContent = `${data.rows_imported} rows imported from ${data.source}; latest SCADA snapshot applied.`;
  } catch (error) { status.textContent = `Import failed: ${error.message}`; }
}
document.querySelector('#usb-import').addEventListener('click', async () => {
  const file = document.querySelector('#hardware-file').files[0];
  if (!file) return document.querySelector('#hardware-status').textContent = 'Select a CSV file first.';
  await importHardware({source: 'usb', csv_text: await file.text()});
});
document.querySelector('#cloud-import').addEventListener('click', async () => {
  const cloudUrl = document.querySelector('#cloud-url').value.trim();
  if (!cloudUrl) return document.querySelector('#hardware-status').textContent = 'Enter a signed HTTPS CSV URL first.';
  await importHardware({source: 'cloud', cloud_url: cloudUrl});
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submit.disabled = true; submit.textContent = 'Assessing…';
  const context = { wind_speed_ms: optionalNumber('wind_speed_ms'), power_output_kw: optionalNumber('power_output_kw') };
  Object.keys(context).forEach(key => context[key] === undefined && delete context[key]);
  const payload = {
    asset_id: value(form, 'asset_id'),
    telemetry: { vibration_mms: number('vibration_mms'), temperature_c: number('temperature_c'), rpm: number('rpm'), oil_viscosity_cst: number('oil_viscosity_cst'), load_pct: number('load_pct') },
    bnn_state: { predicted_rul_days: number('predicted_rul_days'), epistemic_uncertainty: number('epistemic_uncertainty'), aleatoric_uncertainty: number('aleatoric_uncertainty') }
  };
  if (Object.keys(context).length) payload.physics_guided_context = context;
  try {
    const response = await fetch('/api/advisory', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    render(data);
  } catch (error) { showError(`Could not run advisory: ${error.message}`); }
  finally { submit.disabled = false; submit.innerHTML = 'Run advisory <span>→</span>'; }
});
