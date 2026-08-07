"""Anime-inspired cyber HUD for the AeroVigil digital-twin simulator."""

from __future__ import annotations

import math
from html import escape

if __package__:
    from .colors import hex_to_rgba
else:
    from colors import hex_to_rgba


CYBER_TWIN_CSS = r"""
/* ── CYBER PRIME digital-twin experience ───────────────────────── */
.cyber-twin-shell {
  --ct-accent:#00e5a0; --ct-soft:rgba(0,229,160,.18); --ct-hot:#ff3df2;
  position:relative; isolation:isolate; min-height:690px; overflow:hidden;
  border:1px solid rgba(103,232,249,.25); border-radius:28px;
  background:radial-gradient(circle at 50% 18%,rgba(123,44,191,.25),transparent 32%),
    radial-gradient(circle at 12% 65%,var(--ct-soft),transparent 28%),
    linear-gradient(155deg,#030711 0%,#09071b 48%,#020914 100%);
  box-shadow:0 28px 80px rgba(0,0,0,.55),0 0 55px var(--ct-soft),
    inset 0 0 80px rgba(49,46,129,.16); color:#effbff;
  font-family:'Space Grotesk','Inter',system-ui,sans-serif;
}
.cyber-twin-shell::before { content:"";position:absolute;inset:0;z-index:-1;pointer-events:none;
  opacity:.24;background-image:linear-gradient(rgba(103,232,249,.08) 1px,transparent 1px),
  linear-gradient(90deg,rgba(103,232,249,.08) 1px,transparent 1px);background-size:44px 44px;
  mask-image:linear-gradient(to bottom,transparent,#000 30%,#000);
  transform:perspective(500px) rotateX(61deg) scale(1.35) translateY(35%);
  transform-origin:bottom;animation:ct-grid-drift 8s linear infinite; }
.cyber-twin-shell::after { content:"";position:absolute;inset:0;z-index:10;pointer-events:none;
  opacity:.24;mix-blend-mode:screen;background:repeating-linear-gradient(to bottom,transparent 0,
  transparent 3px,rgba(123,236,255,.055) 4px); }
.ct-aurora { position:absolute;inset:-30% -10% auto;height:68%;z-index:-1;filter:blur(45px);
  opacity:.4;background:conic-gradient(from 100deg at 50% 50%,transparent,var(--ct-soft),
  rgba(255,61,242,.16),transparent 64%);animation:ct-aurora 12s ease-in-out infinite alternate; }
.ct-header { position:relative;z-index:4;display:flex;justify-content:space-between;gap:18px;
  align-items:center;padding:20px 24px 15px;border-bottom:1px solid rgba(139,233,255,.13);
  background:linear-gradient(90deg,rgba(2,6,23,.78),rgba(17,12,45,.55),rgba(2,6,23,.78));
  backdrop-filter:blur(18px); }
.ct-eyebrow { margin-bottom:5px;color:var(--ct-accent);font:800 10px 'JetBrains Mono',monospace;
  letter-spacing:.28em;text-transform:uppercase;text-shadow:0 0 16px var(--ct-accent); }
.ct-title { margin:0;font:900 clamp(22px,3vw,35px)/1 'Orbitron','Inter',sans-serif;
  letter-spacing:.04em;background:linear-gradient(90deg,#fff 10%,#8be9ff 52%,var(--ct-accent));
  -webkit-background-clip:text;background-clip:text;color:transparent; }
.ct-title span { color:rgba(178,194,224,.72);font-weight:500; }
.ct-status-cluster { display:flex;gap:9px;flex-wrap:wrap;justify-content:flex-end; }
.ct-chip { display:inline-flex;align-items:center;gap:7px;padding:7px 10px;
  border:1px solid rgba(139,233,255,.18);border-radius:999px;background:rgba(4,12,28,.72);
  color:rgba(222,245,255,.76);font:700 9px 'JetBrains Mono',monospace;
  letter-spacing:.12em;text-transform:uppercase; }
.ct-chip--live { border-color:var(--ct-soft);color:var(--ct-accent); }
.ct-live-dot { width:7px;height:7px;border-radius:50%;background:var(--ct-accent);
  box-shadow:0 0 0 4px var(--ct-soft),0 0 14px var(--ct-accent);animation:ct-pulse 1.8s ease-out infinite; }
.ct-body { position:relative;z-index:2;display:grid;
  grid-template-columns:minmax(170px,.72fr) minmax(330px,1.7fr) minmax(170px,.72fr);
  gap:16px;min-height:535px;padding:18px; }
.ct-hud-column { display:flex;flex-direction:column;gap:11px;z-index:4; }
.ct-hud-card { position:relative;overflow:hidden;padding:13px 14px;
  border:1px solid rgba(139,233,255,.14);border-radius:14px 3px 14px 3px;
  background:linear-gradient(135deg,rgba(4,12,29,.88),rgba(16,11,42,.7));
  box-shadow:inset 3px 0 0 var(--ct-accent),0 10px 28px rgba(0,0,0,.2);backdrop-filter:blur(12px); }
.ct-hud-card::after { content:"";position:absolute;top:0;right:0;width:25px;height:1px;
  background:var(--ct-accent);box-shadow:0 0 12px var(--ct-accent); }
.ct-hud-label { color:rgba(152,188,216,.68);font:700 8px 'JetBrains Mono',monospace;
  letter-spacing:.18em;text-transform:uppercase; }
.ct-hud-value { margin-top:4px;color:#f2fbff;font:800 clamp(18px,2vw,25px) 'Orbitron',sans-serif;
  text-shadow:0 0 18px var(--ct-soft); }
.ct-hud-value small { color:var(--ct-accent);font-size:9px;font-weight:700; }
.ct-meter { height:3px;margin-top:9px;overflow:hidden;border-radius:4px;background:rgba(255,255,255,.07); }
.ct-meter span { display:block;height:100%;border-radius:inherit;
  background:linear-gradient(90deg,#20e3ff,var(--ct-accent));box-shadow:0 0 12px var(--ct-accent);
  animation:ct-meter-in 1.15s cubic-bezier(.2,.8,.2,1) both; }
.ct-stage { position:relative;min-height:500px;overflow:hidden;border:1px solid rgba(139,233,255,.12);
  border-radius:50% 50% 20px 20px/12% 12% 20px 20px;
  background:radial-gradient(circle at 50% 43%,var(--ct-soft),transparent 25%),
  linear-gradient(to bottom,rgba(13,11,42,.36),rgba(1,6,16,.42));
  box-shadow:inset 0 -70px 90px rgba(0,0,0,.45); }
.ct-stage::before { content:"";position:absolute;left:50%;bottom:6%;width:70%;height:18%;
  transform:translateX(-50%);border:1px solid var(--ct-soft);border-radius:50%;
  background:radial-gradient(ellipse,var(--ct-soft),transparent 67%);
  box-shadow:0 0 45px var(--ct-soft),inset 0 0 25px var(--ct-soft);animation:ct-platform 3s ease-in-out infinite; }
.ct-stage::after { content:"";position:absolute;z-index:5;top:-10%;left:5%;width:90%;height:2px;
  opacity:.8;background:linear-gradient(90deg,transparent,var(--ct-accent),#fff,transparent);
  box-shadow:0 0 18px var(--ct-accent);animation:ct-scan 5s linear infinite; }
.ct-moon { position:absolute;top:7%;left:50%;width:210px;height:210px;transform:translateX(-50%);
  border:1px solid rgba(255,61,242,.2);border-radius:50%;opacity:.72;
  background:repeating-linear-gradient(to bottom,transparent 0 11px,rgba(3,7,18,.88) 12px 15px),
  linear-gradient(155deg,rgba(255,122,246,.9),rgba(91,33,182,.4) 63%,transparent 64%);
  box-shadow:0 0 80px rgba(255,61,242,.2); }
.ct-turbine-svg { position:absolute;z-index:3;inset:0;width:100%;height:100%;overflow:visible; }
.ct-rotor { transform-box:fill-box;transform-origin:center;
  animation:ct-rotor-spin var(--ct-rotor-speed,7s) linear infinite;filter:drop-shadow(0 0 6px var(--ct-accent)); }
.ct-orbit { transform-box:fill-box;transform-origin:center;animation:ct-orbit 12s linear infinite; }
.ct-orbit--reverse { animation-direction:reverse;animation-duration:8s; }
.ct-dash { stroke-dasharray:7 12;animation:ct-dash 12s linear infinite; }
.ct-energy-core { transform-box:fill-box;transform-origin:center;animation:ct-core 2.2s ease-in-out infinite; }
.ct-particle { fill:#8be9ff;filter:drop-shadow(0 0 4px #8be9ff);animation:ct-particle 3s ease-in-out infinite; }
.ct-particle:nth-child(2n) { animation-delay:-1.1s; }.ct-particle:nth-child(3n) { animation-delay:-2.2s; }
.ct-callout { position:absolute;z-index:6;display:flex;align-items:center;gap:7px;
  color:rgba(201,239,255,.78);font:700 8px 'JetBrains Mono',monospace;
  letter-spacing:.11em;text-transform:uppercase;animation:ct-float 3.6s ease-in-out infinite; }
.ct-callout::before { content:"";width:22px;height:1px;background:var(--ct-accent);box-shadow:0 0 8px var(--ct-accent); }
.ct-callout--a { top:31%;left:7%; }.ct-callout--b { top:49%;right:5%;animation-delay:-1.8s; }
.ct-callout--c { bottom:19%;left:9%;animation-delay:-.8s; }
.ct-sync-badge { position:absolute;z-index:7;left:50%;bottom:4%;min-width:180px;padding:8px 14px;
  transform:translateX(-50%);border:1px solid var(--ct-soft);border-radius:999px;background:rgba(1,6,18,.82);
  color:var(--ct-accent);font:800 8px 'JetBrains Mono',monospace;letter-spacing:.18em;text-align:center;
  text-transform:uppercase;box-shadow:0 0 24px var(--ct-soft); }
.ct-ai-card { flex:1;min-height:124px;display:grid;grid-template-columns:60px 1fr;gap:9px;align-items:center;
  border-color:rgba(255,61,242,.22);box-shadow:inset 3px 0 0 #ff3df2,0 10px 28px rgba(0,0,0,.2); }
.ct-ai-card--physics { border-color:rgba(32,227,255,.24);box-shadow:inset 3px 0 0 #20e3ff,0 10px 28px rgba(0,0,0,.2); }
.ct-avatar { position:relative;width:56px;height:72px;filter:drop-shadow(0 0 12px rgba(255,61,242,.45)); }
.ct-avatar--physics { filter:drop-shadow(0 0 12px rgba(32,227,255,.48)); }.ct-avatar svg { width:100%;height:100%; }
.ct-ai-name { color:#ff91f8;font:800 11px 'Orbitron',sans-serif;letter-spacing:.08em; }
.ct-ai-card--physics .ct-ai-name { color:#72f2ff; }
.ct-ai-role { display:inline-flex;align-items:center;gap:5px;margin-top:4px;color:rgba(174,205,226,.55);
  font:700 7px 'JetBrains Mono',monospace;letter-spacing:.12em;text-transform:uppercase; }
.ct-ai-role::before { content:"";width:5px;height:5px;border-radius:50%;background:currentColor;
  box-shadow:0 0 8px currentColor;animation:ct-pulse 2.1s ease-out infinite; }
.ct-ai-copy { margin-top:6px;color:rgba(211,221,243,.7);font-size:9px;line-height:1.48; }
.ct-footer { position:relative;z-index:5;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));
  gap:1px;border-top:1px solid rgba(139,233,255,.12);background:rgba(1,5,15,.7); }
.ct-footer-cell { padding:11px 14px;border-right:1px solid rgba(139,233,255,.08); }.ct-footer-cell:last-child{border-right:0}
.ct-footer-label { color:rgba(131,168,198,.58);font:400 7px 'JetBrains Mono',monospace;letter-spacing:.16em;text-transform:uppercase; }
.ct-footer-value { margin-top:3px;color:#dff9ff;font:700 10px 'JetBrains Mono',monospace; }.ct-footer-value strong{color:var(--ct-accent)}
.ct-command-grid { display:grid;grid-template-columns:minmax(0,1.18fr) minmax(0,.82fr);gap:12px;margin:12px 0; }
.ct-command-panel { overflow:hidden;border:1px solid rgba(103,232,249,.15);border-radius:18px;
  background:linear-gradient(145deg,rgba(3,9,24,.96),rgba(17,9,39,.88));
  box-shadow:0 14px 34px rgba(0,0,0,.25),inset 0 0 34px rgba(32,227,255,.03); }
.ct-command-head { display:flex;align-items:center;justify-content:space-between;gap:10px;padding:13px 15px;border-bottom:1px solid rgba(139,233,255,.09); }
.ct-command-head b { color:#edf6ff;font:800 11px 'Orbitron',sans-serif;letter-spacing:.08em; }
.ct-command-head span { color:#00e5a0;font:700 8px 'JetBrains Mono',monospace;letter-spacing:.13em; }
.ct-component-rail { display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;background:rgba(139,233,255,.07); }
.ct-component { position:relative;min-height:122px;padding:13px 10px;background:rgba(3,8,23,.96);text-align:center; }
.ct-component::after { content:"";position:absolute;top:31px;right:-7px;z-index:2;width:14px;height:1px;
  background:linear-gradient(90deg,var(--component-accent),transparent);box-shadow:0 0 7px var(--component-accent); }
.ct-component:last-child::after{display:none}.ct-component-core{width:34px;height:34px;margin:0 auto 9px;border:1px solid var(--component-accent);border-radius:50%;background:radial-gradient(circle,var(--component-soft),transparent 68%);box-shadow:0 0 18px var(--component-soft),inset 0 0 12px var(--component-soft);animation:ct-core 2.5s ease-in-out infinite}
.ct-component-name{color:#dff8ff;font:800 8px 'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase}.ct-component-score{margin-top:5px;color:var(--component-accent);font:800 16px 'Orbitron',sans-serif}.ct-component-state{margin-top:3px;color:rgba(160,191,214,.5);font:600 7px 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase}.ct-component-bar{height:2px;margin-top:9px;background:rgba(255,255,255,.06)}.ct-component-bar i{display:block;height:100%;background:var(--component-accent);box-shadow:0 0 8px var(--component-accent);animation:ct-meter-in 1s ease-out both}
.ct-council-stream{display:flex;flex-direction:column;gap:8px;padding:12px}.ct-council-msg{display:grid;grid-template-columns:34px 1fr;gap:9px;padding:9px;border:1px solid rgba(139,233,255,.08);border-radius:11px;background:rgba(2,7,20,.68)}.ct-council-avatar{display:flex;align-items:center;justify-content:center;width:32px;height:32px;border:1px solid currentColor;border-radius:9px;color:#ff91f8;font:900 11px 'Orbitron',sans-serif;box-shadow:0 0 12px rgba(255,61,242,.13)}.ct-council-msg--kai .ct-council-avatar{color:#72f2ff}.ct-council-msg--synth .ct-council-avatar{color:#00e5a0}.ct-council-name{color:rgba(171,209,229,.56);font:700 7px 'JetBrains Mono',monospace;letter-spacing:.12em;text-transform:uppercase}.ct-council-copy{margin-top:3px;color:rgba(226,238,249,.78);font-size:10px;line-height:1.45}
.ct-evidence-chips{display:flex;gap:5px;flex-wrap:wrap;padding:0 12px 12px}.ct-evidence-chip{padding:4px 7px;border:1px solid rgba(139,233,255,.12);border-radius:999px;color:rgba(162,207,229,.64);font:700 7px 'JetBrains Mono',monospace;letter-spacing:.07em;text-transform:uppercase}.ct-review-receipt{padding:15px;border:1px solid rgba(0,229,160,.18);border-radius:15px;background:linear-gradient(135deg,rgba(0,229,160,.055),rgba(3,9,24,.92))}.ct-review-receipt strong{color:#00e5a0;font:800 11px 'Orbitron',sans-serif;letter-spacing:.08em}.ct-review-receipt p{margin:7px 0 0;color:rgba(205,228,239,.7);font-size:11px;line-height:1.55}.ct-review-meta{margin-top:9px;color:rgba(139,189,212,.48);font:700 8px 'JetBrains Mono',monospace;letter-spacing:.09em;text-transform:uppercase}
.ct-compare-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin:10px 0 14px}.ct-compare-card{padding:12px;border:1px solid var(--compare-soft);border-radius:13px;background:linear-gradient(145deg,rgba(4,10,27,.94),rgba(14,8,34,.86));box-shadow:inset 3px 0 0 var(--compare-accent)}.ct-compare-card b{color:#edf6ff;font-size:11px}.ct-compare-value{margin-top:7px;color:var(--compare-accent);font:800 19px 'Orbitron',sans-serif}.ct-compare-meta{margin-top:4px;color:rgba(157,190,212,.56);font:700 7px 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase}
.agent-mesh{display:grid;grid-template-columns:minmax(190px,.8fr) minmax(420px,2fr) auto;gap:16px;align-items:center;margin:9px 0 12px;padding:11px 14px;overflow:hidden;border:1px solid rgba(103,232,249,.15);border-radius:14px;background:linear-gradient(90deg,rgba(255,61,242,.045),rgba(3,10,27,.88),rgba(32,227,255,.045));box-shadow:inset 0 0 28px rgba(32,227,255,.035)}.agent-mesh-title{display:flex;align-items:center;gap:10px;color:#edf6ff;font:700 9px 'JetBrains Mono',monospace;letter-spacing:.1em}.agent-mesh-title span:last-child{display:flex;flex-direction:column;gap:2px}.agent-mesh-title small{color:rgba(152,188,216,.5);font-size:7px;font-weight:500;letter-spacing:.04em}.agent-mesh-pulse{width:8px;height:8px;border-radius:50%;background:#ff3df2;box-shadow:0 0 0 4px rgba(255,61,242,.1),0 0 13px #ff3df2;animation:ct-pulse 2s infinite}.agent-mesh-path{display:flex;align-items:center;min-width:0}.agent-mesh-node{flex:none;display:flex;flex-direction:column;gap:1px;align-items:center;color:#8be9ff;font:800 8px 'JetBrains Mono',monospace;letter-spacing:.08em}.agent-mesh-node small{color:rgba(155,188,211,.46);font-size:6px;font-weight:500}.agent-mesh-line{flex:1;min-width:12px;height:1px;margin:0 5px;background:linear-gradient(90deg,#ff3df2,#20e3ff);box-shadow:0 0 7px rgba(32,227,255,.6);position:relative}.agent-mesh-line::after{content:"";position:absolute;top:-2px;width:5px;height:5px;border-radius:50%;background:#fff;box-shadow:0 0 8px #20e3ff;animation:ct-mesh-packet 2.4s linear infinite}.agent-mesh-state{color:#00e5a0;font:800 8px 'JetBrains Mono',monospace;letter-spacing:.12em;white-space:nowrap}
.cyber-control-panel{border:1px solid rgba(103,232,249,.16)!important;border-radius:20px!important;background:linear-gradient(145deg,rgba(4,9,24,.94),rgba(16,10,38,.88))!important;box-shadow:inset 0 0 45px rgba(59,130,246,.05),0 16px 38px rgba(0,0,0,.24)!important;padding:16px!important}.cyber-result-panel{min-height:100%;border:1px solid rgba(103,232,249,.14);border-radius:18px;overflow:hidden;background:rgba(3,8,23,.86)}.ct-result-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:14px 16px;border-bottom:1px solid rgba(139,233,255,.1)}.ct-result-kicker{color:rgba(142,188,218,.6);font:400 8px 'JetBrains Mono',monospace;letter-spacing:.18em;text-transform:uppercase}.ct-result-risk{color:var(--result-accent);font:900 13px 'Orbitron',sans-serif;text-shadow:0 0 15px var(--result-soft)}.ct-result-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:rgba(139,233,255,.08)}.ct-result-stat{padding:14px;background:rgba(3,8,23,.96)}.ct-result-stat span{display:block;color:rgba(142,188,218,.58);font:400 7px 'JetBrains Mono',monospace;letter-spacing:.13em;text-transform:uppercase}.ct-result-stat strong{display:block;margin-top:5px;color:#effbff;font:700 17px 'Orbitron',sans-serif}.ct-result-track{padding:14px 16px}.ct-result-track-title{margin-bottom:10px;color:rgba(187,222,240,.68);font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}.ct-result-steps{display:flex;align-items:center;gap:5px}.ct-result-step{flex:1;height:5px;border-radius:8px;background:rgba(255,255,255,.07);overflow:hidden}.ct-result-step.on{background:var(--result-soft);box-shadow:0 0 12px var(--result-soft)}.ct-result-step.on::after{content:"";display:block;width:100%;height:100%;background:var(--result-accent);animation:ct-meter-in .8s ease-out both}
@keyframes ct-grid-drift{to{background-position:0 44px,44px 0}}@keyframes ct-aurora{to{transform:translate3d(8%,8%,0) rotate(8deg) scale(1.08);opacity:.62}}@keyframes ct-pulse{0%{box-shadow:0 0 0 0 var(--ct-soft),0 0 12px var(--ct-accent)}75%,100%{box-shadow:0 0 0 9px transparent,0 0 20px var(--ct-accent)}}@keyframes ct-meter-in{from{width:0;filter:brightness(2)}}@keyframes ct-platform{50%{transform:translateX(-50%) scale(1.05);opacity:.68}}@keyframes ct-scan{from{transform:translateY(0)}to{transform:translateY(590px)}}@keyframes ct-rotor-spin{to{transform:rotate(360deg)}}@keyframes ct-orbit{to{transform:rotate(360deg)}}@keyframes ct-dash{to{stroke-dashoffset:-190}}@keyframes ct-core{50%{transform:scale(1.28);opacity:.62}}@keyframes ct-particle{0%,100%{opacity:.15;transform:translateY(8px)}50%{opacity:1;transform:translateY(-9px)}}@keyframes ct-float{50%{transform:translateY(-7px);opacity:.72}}@keyframes ct-mesh-packet{from{left:0}to{left:calc(100% - 5px)}}
@media(max-width:900px){.agent-mesh,.ct-command-grid{grid-template-columns:1fr}.agent-mesh-state{display:none}.cyber-twin-shell{min-height:auto}.ct-body{grid-template-columns:1fr 1fr}.ct-stage{grid-column:1/-1;grid-row:1;min-height:470px}.ct-hud-column{grid-row:2}}
@media(max-width:620px){.ct-header{align-items:flex-start;flex-direction:column}.ct-status-cluster{justify-content:flex-start}.ct-body{display:flex;flex-direction:column}.ct-stage{min-height:430px;order:-1}.ct-hud-column{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}.ct-ai-card{grid-column:1/-1}.ct-footer{grid-template-columns:repeat(2,minmax(0,1fr))}.ct-component-rail{grid-template-columns:repeat(2,minmax(0,1fr))}.ct-component::after{display:none}}
@media(prefers-reduced-motion:reduce){.cyber-twin-shell *,.cyber-twin-shell::before,.cyber-twin-shell::after{animation-duration:.001ms!important;animation-iteration-count:1!important}}
"""


def _finite(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else fallback


def _pct(value: float) -> float:
    return min(100.0, max(0.0, _finite(value)))


def _hud_card(label: str, value: str, unit: str, percentage: float) -> str:
    return f"""<div class="ct-hud-card"><div class="ct-hud-label">{label}</div>
    <div class="ct-hud-value">{value} <small>{unit}</small></div>
    <div class="ct-meter"><span style="width:{_pct(percentage):.1f}%"></span></div></div>"""


def _avatar(kind: str) -> str:
    if kind == "mika":
        return """<svg viewBox="0 0 70 90" aria-hidden="true"><path d="M14 30Q17 8 35 5Q56 9 58 31L53 59Q45 76 35 78Q22 73 16 57Z" fill="#120c2b" stroke="#ff65f5"/><path d="M14 31Q9 16 23 9Q35-1 53 13L62 28L54 23L49 33L43 18L31 31L25 18Z" fill="#ff3df2" fill-opacity=".55" stroke="#ff9df8"/><path d="M18 38Q35 29 53 38L50 52Q35 58 20 51Z" fill="#061b2d" stroke="#8be9ff"/><path d="M24 43L32 41M40 41L48 43" stroke="#8be9ff" stroke-width="2.5"/></svg>"""
    return """<svg viewBox="0 0 70 90" aria-hidden="true"><path d="M13 29L20 10L35 4L51 11L59 30L54 60L43 76L27 76L16 59Z" fill="#07172c" stroke="#20e3ff"/><path d="M13 29L7 24L14 12L26 5L24 20L35 10L47 20L45 5L58 14L64 27L57 30L50 23L20 23Z" fill="#20e3ff" fill-opacity=".28" stroke="#8be9ff"/><path d="M16 35L54 35L51 50Q35 56 19 50Z" fill="#020b19" stroke="#20e3ff"/><path d="M22 42L31 40M39 40L48 42" stroke="#72f2ff" stroke-width="2.5"/></svg>"""


def render_cyber_twin(
    scenario: str,
    vibration: float,
    bearing_temp: float,
    generator_temp: float,
    power: float,
    wind: float,
    operating_hours: float,
    rul_days: float,
    risk: str,
    accent: str,
) -> str:
    """Render the animated advisory-only digital-twin HUD."""
    safe_scenario = escape(str(scenario))
    vibration, bearing_temp = _finite(vibration), _finite(bearing_temp)
    generator_temp, power = _finite(generator_temp), _finite(power)
    wind, operating_hours, rul_days = _finite(wind), _finite(operating_hours), _finite(rul_days)
    soft = hex_to_rgba(accent, 0.18)  # validates a Plotly/CSS-safe opaque accent
    load_pct = _pct(power / 2500.0 * 100.0)
    vib_pct = _pct(vibration / 45.0 * 100.0)
    bearing_pct = _pct((bearing_temp - 20.0) / 110.0 * 100.0)
    generator_pct = _pct((generator_temp - 25.0) / 145.0 * 100.0)
    life_pct = _pct(rul_days / 365.0 * 100.0)
    stress_pct = _pct(vib_pct * 0.42 + bearing_pct * 0.28 + generator_pct * 0.2 + load_pct * 0.1)
    sync_pct = _pct(99.7 - stress_pct * 0.025)
    rotor_rpm = max(2.4, min(18.0, wind * 0.92))
    torque_knm = power / max(rotor_rpm * math.tau / 60.0, 0.1)
    wear_pct = _pct(100.0 - life_pct)
    rotor_speed = max(2.6, min(11.0, 14.5 - wind * 0.65))
    risk = escape(str(risk).upper())

    from src.agents.cyber_team import build_cyber_team_brief

    team = build_cyber_team_brief(
        asset_id="TWIN-07",
        predicted_rul_days=rul_days,
        cumulative_wear=wear_pct / 100.0,
        telemetry={
            "vibration_mms": vibration,
            "temperature_c": bearing_temp,
            "power_output": power,
        },
        risk=risk,
    )
    mika = team["agents"]["mika"]["finding"]
    kai = team["agents"]["kai"]["finding"]
    left_cards = "".join(
        [
            _hud_card("Rotor velocity", f"{rotor_rpm:.1f}", "RPM", rotor_rpm / 18 * 100),
            _hud_card("Bearing thermal", f"{bearing_temp:.1f}", "°C", bearing_pct),
            _hud_card("Nacelle vibration", f"{vibration:.1f}", "MM/S", vib_pct),
            _hud_card("Generator core", f"{generator_temp:.0f}", "°C", generator_pct),
        ]
    )
    return f"""
    <section class="cyber-twin-shell" style="--ct-accent:{accent};--ct-soft:{soft};--ct-rotor-speed:{rotor_speed:.2f}s;">
      <div class="ct-aurora"></div><header class="ct-header"><div><div class="ct-eyebrow">AeroVigil // Neural mirror online</div>
      <h2 class="ct-title">CYBER PRIME <span>・TWIN 07 // DUAL AGENT</span></h2></div><div class="ct-status-cluster">
      <div class="ct-chip ct-chip--live"><span class="ct-live-dot"></span>synchronized {sync_pct:.2f}%</div>
      <div class="ct-chip">2 agents · {team["agreement_score_pct"]:.1f}% consensus</div><div class="ct-chip">{risk} state</div></div></header>
      <div class="ct-body"><aside class="ct-hud-column">{left_cards}</aside><div class="ct-stage"><div class="ct-moon"></div>
      <div class="ct-callout ct-callout--a">physics layer</div><div class="ct-callout ct-callout--b">BNN probability field</div>
      <div class="ct-callout ct-callout--c">ISO 281 bearing core</div>
      <svg class="ct-turbine-svg" viewBox="0 0 520 560" role="img" aria-label="Animated holographic wind turbine digital twin">
      <defs><linearGradient id="ctTower" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e9fbff" stop-opacity=".9"/><stop offset=".45" stop-color="{accent}" stop-opacity=".72"/><stop offset="1" stop-color="#6d28d9" stop-opacity=".18"/></linearGradient><linearGradient id="ctBlade"><stop offset="0" stop-color="#fff"/><stop offset=".52" stop-color="#8be9ff" stop-opacity=".65"/><stop offset="1" stop-color="{accent}" stop-opacity=".14"/></linearGradient></defs>
      <g opacity=".75"><circle class="ct-particle" cx="88" cy="115" r="2"/><circle class="ct-particle" cx="423" cy="147" r="1.5"/><circle class="ct-particle" cx="455" cy="324" r="2"/><circle class="ct-particle" cx="72" cy="369" r="1.5"/></g>
      <g class="ct-orbit" fill="none" stroke="{accent}" stroke-opacity=".25"><ellipse class="ct-dash" cx="260" cy="229" rx="169" ry="72" transform="rotate(-12 260 229)"/><ellipse cx="260" cy="229" rx="126" ry="52" stroke-dasharray="2 9"/></g>
      <g class="ct-orbit ct-orbit--reverse" fill="none" stroke="#ff3df2" stroke-opacity=".22"><ellipse class="ct-dash" cx="260" cy="229" rx="190" ry="90" transform="rotate(17 260 229)"/></g>
      <path d="M224 500L245 231L275 231L299 500Z" fill="url(#ctTower)"/><path d="M245 251L238 483M274 251L286 483M248 310L277 310M242 390L284 390" fill="none" stroke="#dffbff" stroke-opacity=".34"/>
      <path d="M219 501Q260 481 304 501L319 517L203 517Z" fill="{accent}" fill-opacity=".23" stroke="{accent}"/><rect x="237" y="213" width="75" height="29" rx="13" fill="#07152b" stroke="{accent}" stroke-width="2"/>
      <g class="ct-rotor"><path d="M260 229C246 178 220 103 244 42C263 112 267 176 260 229Z" fill="url(#ctBlade)" stroke="#c9f7ff"/><path d="M260 229C311 222 390 224 431 274C357 269 299 249 260 229Z" fill="url(#ctBlade)" stroke="#c9f7ff"/><path d="M260 229C224 266 174 328 110 338C153 277 207 243 260 229Z" fill="url(#ctBlade)" stroke="#c9f7ff"/></g>
      <g class="ct-energy-core"><circle cx="260" cy="229" r="20" fill="#031222" stroke="{accent}" stroke-width="3"/><circle cx="260" cy="229" r="9" fill="{accent}"/><circle cx="260" cy="229" r="3" fill="#fff"/></g></svg>
      <div class="ct-sync-badge">{safe_scenario} // forecast linked</div></div>
      <aside class="ct-hud-column">{_hud_card("Projected life", f"{rul_days:.0f}", "DAYS", life_pct)}
      {_hud_card("Active power", f"{power / 1000:.2f}", "MW", load_pct)}{_hud_card("Rotor torque", f"{torque_knm:.0f}", "kN·m", torque_knm / 2200 * 100)}
      <div class="ct-hud-card ct-ai-card"><div class="ct-avatar">{_avatar("mika")}</div><div><div class="ct-ai-name">MIKA // AI</div><div class="ct-ai-role">Maintenance strategist</div><div class="ct-ai-copy">{mika}</div></div></div>
      <div class="ct-hud-card ct-ai-card ct-ai-card--physics"><div class="ct-avatar ct-avatar--physics">{_avatar("kai")}</div><div><div class="ct-ai-name">KAI // PHYSICS</div><div class="ct-ai-role">Constraint sentinel</div><div class="ct-ai-copy">{kai}</div></div></div></aside></div>
      <footer class="ct-footer"><div class="ct-footer-cell"><div class="ct-footer-label">Simulation clock</div><div class="ct-footer-value">{operating_hours:,.0f} <strong>H</strong></div></div><div class="ct-footer-cell"><div class="ct-footer-label">Wind vector</div><div class="ct-footer-value">{wind:.1f} <strong>M/S</strong></div></div><div class="ct-footer-cell"><div class="ct-footer-label">Stress index</div><div class="ct-footer-value">{stress_pct:.1f} <strong>/ 100</strong></div></div><div class="ct-footer-cell"><div class="ct-footer-label">Life consumed</div><div class="ct-footer-value">{wear_pct:.1f}<strong>%</strong></div></div><div class="ct-footer-cell"><div class="ct-footer-label">Agent evidence mesh</div><div class="ct-footer-value">{len(team["connected_sources"])} <strong>POINTS LINKED</strong></div></div></footer>
    </section>"""


def render_twin_result(
    scenario: str,
    risk: str,
    accent: str,
    final_rul: float,
    total_hours: float,
    degradation: float,
    stress_pct: float,
) -> str:
    """Render the simulation mission-result panel."""
    safe_scenario, risk = escape(str(scenario)), escape(str(risk).upper())
    soft = hex_to_rgba(accent, 0.2)
    active_steps = 1 if risk == "LOW" else 2 if risk == "MODERATE" else 3 if risk == "HIGH" else 4
    steps = "".join(
        f'<span class="ct-result-step{" on" if index < active_steps else ""}"></span>'
        for index in range(4)
    )
    return f"""<div class="cyber-result-panel" style="--result-accent:{accent};--result-soft:{soft};"><div class="ct-result-head"><div><div class="ct-result-kicker">MIKA + KAI // dual-agent consensus complete</div><div style="color:#effbff;font-weight:800;margin-top:3px;">{safe_scenario}</div></div><div class="ct-result-risk">{risk}</div></div><div class="ct-result-grid"><div class="ct-result-stat"><span>Final healthy life</span><strong>{final_rul:.0f} days</strong></div><div class="ct-result-stat"><span>Simulation clock</span><strong>{total_hours:,.0f} h</strong></div><div class="ct-result-stat"><span>Degradation vector</span><strong>{degradation:.2f}×</strong></div><div class="ct-result-stat"><span>Stress load</span><strong>{_pct(stress_pct):.1f}%</strong></div></div><div class="ct-result-track"><div class="ct-result-track-title">Escalation horizon // monitor → plan → intervene → urgent</div><div class="ct-result-steps">{steps}</div></div></div>"""


def render_component_diagnostics(
    vibration: float,
    bearing_temp: float,
    generator_temp: float,
    power: float,
    wind: float,
    rul_days: float,
) -> str:
    """Render an exploded component health scan from projected twin signals."""
    vibration, bearing_temp = _finite(vibration), _finite(bearing_temp)
    generator_temp, power, wind, rul_days = map(_finite, (generator_temp, power, wind, rul_days))
    load = _pct(power / 2500.0 * 100.0)
    components = [
        ("Rotor", 100.0 - abs(wind - 10.0) * 2.5 - vibration * 0.35),
        ("Main bearing", 100.0 - vibration * 1.45 - max(0.0, bearing_temp - 55.0) * 0.72),
        ("Gearbox", 100.0 - vibration * 1.15 - max(0.0, load - 78.0) * 0.36),
        ("Generator", 100.0 - max(0.0, generator_temp - 65.0) * 0.7 - max(0.0, load - 90.0) * 0.3),
        ("Converter", 100.0 - max(0.0, load - 72.0) * 0.48 - max(0.0, 45.0 - rul_days) * 0.2),
    ]
    cards = []
    for index, (name, raw_score) in enumerate(components, start=1):
        score = _pct(raw_score)
        state, accent = (
            ("nominal", "#00e5a0")
            if score >= 78
            else ("watch", "#ffd600")
            if score >= 58
            else ("stressed", "#ff6d00")
            if score >= 38
            else ("critical", "#ff1744")
        )
        cards.append(
            f"""<div class="ct-component" style="--component-accent:{accent};--component-soft:{hex_to_rgba(accent, 0.16)};"><div class="ct-component-core"></div><div class="ct-component-name">{index:02d} · {name}</div><div class="ct-component-score">{score:.0f}</div><div class="ct-component-state">{state}</div><div class="ct-component-bar"><i style="width:{score:.1f}%"></i></div></div>"""
        )
    weakest = min(components, key=lambda item: item[1])[0]
    return f"""<section class="ct-command-panel"><div class="ct-command-head"><b>COMPONENT RESONANCE SCAN</b><span>5 SUBSYSTEMS // LIVE X-RAY</span></div><div class="ct-component-rail">{"".join(cards)}</div><div style="padding:9px 13px;color:rgba(158,194,215,.58);font:700 8px monospace;letter-spacing:.09em;text-transform:uppercase;">KAI focus node · {escape(weakest)} // scores are scenario-relative health indicators</div></section>"""


def render_agent_council(
    scenario: str,
    vibration: float,
    bearing_temp: float,
    power: float,
    rul_days: float,
    risk: str,
) -> str:
    """Render the MIKA/KAI evidence conversation and human handoff."""
    from src.agents.cyber_team import build_cyber_team_brief

    team = build_cyber_team_brief(
        asset_id="TWIN-07",
        predicted_rul_days=rul_days,
        cumulative_wear=max(0.0, min(1.0, 1.0 - _finite(rul_days) / 450.0)),
        telemetry={
            "vibration_mms": vibration,
            "temperature_c": bearing_temp,
            "power_output": power,
        },
        risk=risk,
    )
    mika, kai = team["agents"]["mika"]["finding"], team["agents"]["kai"]["finding"]
    chips = "".join(
        f'<span class="ct-evidence-chip">{escape(source.replace("_", " "))}</span>'
        for source in team["connected_sources"]
    )
    return f"""<section class="ct-command-panel"><div class="ct-command-head"><b>DUAL-AGENT COUNCIL</b><span>{team["agreement_score_pct"]:.1f}% AGREEMENT</span></div><div class="ct-council-stream"><div class="ct-council-msg"><div class="ct-council-avatar">M</div><div><div class="ct-council-name">MIKA · Maintenance strategy</div><div class="ct-council-copy">{mika}</div></div></div><div class="ct-council-msg ct-council-msg--kai"><div class="ct-council-avatar">K</div><div><div class="ct-council-name">KAI · Physics challenge</div><div class="ct-council-copy">{kai}</div></div></div><div class="ct-council-msg ct-council-msg--synth"><div class="ct-council-avatar">Σ</div><div><div class="ct-council-name">Synthesis · Human gate required</div><div class="ct-council-copy">{team["shared_summary"]} Review within approximately {team["review_window_days"]:.1f} days.</div></div></div></div><div class="ct-evidence-chips">{chips}</div><div style="padding:0 12px 12px;color:rgba(158,194,215,.48);font:700 8px monospace;letter-spacing:.08em;">SCENARIO THREAD · {escape(str(scenario))} // ADVISORY ONLY</div></section>"""


def render_human_review_receipt(decision: str, scenario: str) -> str:
    """Create a non-actuating audit receipt for an operator review decision."""
    outcomes = {
        "Acknowledge evidence": (
            "EVIDENCE ACKNOWLEDGED",
            "The agent brief has been marked as reviewed. Continue monitoring under the approved human workflow.",
        ),
        "Request engineering review": (
            "ENGINEERING REVIEW REQUESTED",
            "The evidence package is ready for a qualified reliability engineer to assess against OEM guidance.",
        ),
        "Escalate to reliability lead": (
            "RELIABILITY LEAD ESCALATION",
            "The advisory has been elevated for priority human review; no turbine command has been issued.",
        ),
    }
    if decision not in outcomes:
        raise ValueError(f"unknown human review decision: {decision}")
    title, copy = outcomes[decision]
    return f"""<div class="ct-review-receipt"><strong>HUMAN GATE // {title}</strong><p>{copy}</p><div class="ct-review-meta">Scenario · {escape(str(scenario))} // decision-support only // audit state recorded</div></div>"""


def render_scenario_comparison(rows: list[dict]) -> str:
    """Render ranked scenario cards beneath the comparison chart."""
    if not rows:
        return '<div class="ct-review-receipt"><strong>SELECT SCENARIOS</strong><p>Choose at least one future to compare.</p></div>'
    cards = []
    for rank, row in enumerate(
        sorted(rows, key=lambda item: item["final_rul"], reverse=True), start=1
    ):
        accent = str(row["accent"])
        cards.append(
            f"""<div class="ct-compare-card" style="--compare-accent:{accent};--compare-soft:{hex_to_rgba(accent, 0.18)};"><b>{rank:02d} · {escape(str(row["scenario"]))}</b><div class="ct-compare-value">{row["final_rul"]:.0f}d</div><div class="ct-compare-meta">wear {row["final_wear"]:.1f}% · stress {row["stress_pct"]:.1f}%</div><div class="ct-compare-meta">simulated energy {row["energy_mwh"]:,.0f} MWh · {escape(str(row["risk"]))}</div></div>"""
        )
    return f"""<div class="ct-compare-summary">{"".join(cards)}</div><div style="color:rgba(150,185,208,.48);font:700 8px monospace;letter-spacing:.08em;text-transform:uppercase;">Ranking maximizes projected healthy-life runway. Energy is an illustrative scenario estimate, not a commercial forecast.</div>"""


def render_agent_answer(question: str, answer: str, agent: str, evidence: list[str]) -> str:
    """Render a grounded answer from the deterministic agent council."""
    agent = str(agent).upper()
    accent = "#72f2ff" if agent == "KAI" else "#ff91f8" if agent == "MIKA" else "#00e5a0"
    chips = "".join(
        f'<span class="ct-evidence-chip">{escape(str(item).replace("_", " "))}</span>'
        for item in evidence
    )
    return f"""<section class="ct-command-panel" style="margin-top:10px;"><div class="ct-command-head"><b>OPERATOR QUESTION</b><span>GROUNDED RESPONSE</span></div><div style="padding:12px 14px;border-bottom:1px solid rgba(139,233,255,.08);color:rgba(207,226,239,.68);font-size:11px;">“{escape(str(question))}”</div><div class="ct-council-stream"><div class="ct-council-msg"><div class="ct-council-avatar" style="color:{accent};">{escape(agent[:1] or "Σ")}</div><div><div class="ct-council-name" style="color:{accent};">{escape(agent)} // evidence response</div><div class="ct-council-copy">{escape(str(answer))}</div></div></div></div><div class="ct-evidence-chips">{chips}</div><div style="padding:0 12px 12px;color:rgba(158,194,215,.45);font:700 8px monospace;">HUMAN REVIEW REQUIRED · NO CONTROL COMMAND GENERATED</div></section>"""
