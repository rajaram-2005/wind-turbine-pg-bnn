import 'dart:convert';
import 'package:flutter/material.dart';

import '../services/api_service.dart';

/// Developer screen: submit manual JSON payloads directly to `/api/model`
/// with gauge + chart interpretation of the RUL result.
class LowLevelInferenceScreen extends StatefulWidget {
  const LowLevelInferenceScreen({super.key, required this.api});
  final ApiService api;

  @override
  State<LowLevelInferenceScreen> createState() => _LowLevelInferenceScreenState();
}

class _LowLevelInferenceScreenState extends State<LowLevelInferenceScreen> {
  final _controller = TextEditingController(
    text: const JsonEncoder.withIndent('  ').convert({
      'vibration_rms': 3.2,
      'bearing_temp': 62.0,
      'generator_temp': 88.0,
      'power_output': 1850.0,
      'wind_speed': 11.5,
      'operating_hours': 42000.0,
      'n_mcmc_samples': 100,
    }),
  );
  bool _busy = false;
  Map<String, dynamic>? _respMap;
  String? _rawResponse;
  bool _error = false;

  Future<void> _submit() async {
    setState(() {
      _busy = true;
      _rawResponse = null;
      _respMap = null;
      _error = false;
    });
    try {
      final payload = jsonDecode(_controller.text) as Map<String, dynamic>;
      final resp = await widget.api.postModelInference(payload);
      setState(() {
        _respMap = resp;
        _rawResponse = const JsonEncoder.withIndent('  ').convert(resp);
      });
    } catch (e) {
      setState(() {
        _error = true;
        _rawResponse = e.toString();
      });
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  double? _asDouble(dynamic v) => v is num ? v.toDouble() : double.tryParse('$v');

  @override
  Widget build(BuildContext context) {
    final meanRul = _respMap != null ? _asDouble(_respMap!['predicted_rul_days'] ?? _respMap!['mean_rul_days'] ?? _respMap!['mean']) : null;
    final epi = _respMap != null ? _asDouble(_respMap!['epistemic_std'] ?? _respMap!['epistemic_uncertainty'] ?? _respMap!['uncertainty_days']) : null;
    final ale = _respMap != null ? _asDouble(_respMap!['aleatoric_std'] ?? _respMap!['aleatoric_uncertainty']) : null;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Low-Level Inference', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 4),
          const Text('POST a raw JSON payload directly to /api/model – the same six-signal SCADA schema used by hardware_agent.'),
          const SizedBox(height: 16),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                flex: 2,
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: SizedBox(
                      height: 420,
                      child: TextField(
                        controller: _controller,
                        maxLines: null,
                        expands: true,
                        textAlignVertical: TextAlignVertical.top,
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 13),
                        decoration: const InputDecoration(border: InputBorder.none, hintText: 'JSON payload'),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                flex: 3,
                child: Column(
                  children: [
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Result', style: Theme.of(context).textTheme.titleSmall),
                            const SizedBox(height: 8),
                            if (_respMap != null && meanRul != null)
                              Wrap(
                                spacing: 12,
                                runSpacing: 12,
                                children: [
                                  _MetricChip(label: 'RUL', value: '${meanRul.toStringAsFixed(1)} d', color: const Color(0xFF2DD4BF)),
                                  if (epi != null) _MetricChip(label: 'Epistemic σ', value: '${epi.toStringAsFixed(2)}', color: const Color(0xFF60A5FA)),
                                  if (ale != null) _MetricChip(label: 'Aleatoric σ', value: '${ale.toStringAsFixed(2)}', color: const Color(0xFFF59E0B)),
                                ],
                              ),
                            const SizedBox(height: 12),
                            SingleChildScrollView(
                              child: SelectableText(
                                _rawResponse ?? 'Response will appear here.',
                                style: TextStyle(fontFamily: 'monospace', fontSize: 12, color: _error ? const Color(0xFFEF4444) : null),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    if (_respMap != null && meanRul != null)
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('Gauge interpretation', style: Theme.of(context).textTheme.titleSmall),
                              const SizedBox(height: 12),
                              _RulGauge(rul: meanRul),
                              const SizedBox(height: 12),
                              _ResultTable(resp: _respMap!),
                            ],
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _busy ? null : _submit,
            icon: _busy ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.send),
            label: const Text('Run Inference'),
          ),
        ],
      ),
    );
  }
}

class _MetricChip extends StatelessWidget {
  const _MetricChip({required this.label, required this.value, required this.color});
  final String label;
  final String value;
  final Color color;
  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(Icons.circle, size: 10, color: color),
      label: Text('$label: $value', style: const TextStyle(fontSize: 12)),
    );
  }
}

class _RulGauge extends StatelessWidget {
  const _RulGauge({required this.rul});
  final double rul;
  @override
  Widget build(BuildContext context) {
    final frac = (rul / 365).clamp(0.0, 1.0);
    Color col = Color.lerp(const Color(0xFFEF4444), const Color(0xFF22C55E), frac)!;
    return SizedBox(
      height: 100,
      child: Row(
        children: [
          SizedBox(width: 80, height: 80, child: CircularProgressIndicator(value: frac, strokeWidth: 8, backgroundColor: Colors.white10, valueColor: AlwaysStoppedAnimation(col))),
          const SizedBox(width: 16),
          Expanded(child: Text('${rul.toStringAsFixed(1)} days RUL – ${frac < 0.2 ? 'Critical' : frac < 0.4 ? 'High' : frac < 0.7 ? 'Watch' : 'Healthy'}', style: Theme.of(context).textTheme.titleMedium)),
        ],
      ),
    );
  }
}

class _ResultTable extends StatelessWidget {
  const _ResultTable({required this.resp});
  final Map<String, dynamic> resp;
  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: DataTable(
        columns: const [DataColumn(label: Text('Field')), DataColumn(label: Text('Value'))],
        rows: [
          for (final e in resp.entries.take(20))
            DataRow(cells: [
              DataCell(Text('${e.key}', style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12))),
              DataCell(SelectableText('${e.value}', style: const TextStyle(fontSize: 12, fontFamily: 'monospace'))),
            ]),
        ],
      ),
    );
  }
}
