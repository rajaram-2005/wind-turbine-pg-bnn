import 'dart:convert';
import 'package:flutter/material.dart';

import '../services/api_service.dart';

/// Developer screen: submit manual JSON payloads directly to `/api/model`.
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
  String? _response;
  bool _error = false;

  Future<void> _submit() async {
    setState(() {
      _busy = true;
      _response = null;
      _error = false;
    });
    try {
      final payload = jsonDecode(_controller.text) as Map<String, dynamic>;
      final resp = await widget.api.postModelInference(payload);
      setState(() => _response = const JsonEncoder.withIndent('  ').convert(resp));
    } catch (e) {
      setState(() {
        _error = true;
        _response = e.toString();
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

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Low-Level Inference', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 4),
          const Text('POST a raw JSON payload directly to /api/model.'),
          const SizedBox(height: 16),
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: TextField(
                        controller: _controller,
                        maxLines: null,
                        expands: true,
                        textAlignVertical: TextAlignVertical.top,
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 13),
                        decoration: const InputDecoration(
                          border: InputBorder.none,
                          hintText: 'JSON payload',
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: SingleChildScrollView(
                        child: SelectableText(
                          _response ?? 'Response will appear here.',
                          style: TextStyle(
                            fontFamily: 'monospace',
                            fontSize: 13,
                            color: _error ? const Color(0xFFEF4444) : null,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _busy ? null : _submit,
            icon: _busy
                ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.send),
            label: const Text('Run Inference'),
          ),
        ],
      ),
    );
  }
}
