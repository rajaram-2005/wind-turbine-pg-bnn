import 'package:flutter/material.dart';

import '../services/api_service.dart';

/// MIKA + KAI Agent Copilot: ask the dual-agent council, record advisory-only
/// human decisions at the decision gate, and compare parallel scenario futures.
/// Restored from the legacy Cyber Twin experience on the canonical API.
class AgentsScreen extends StatefulWidget {
  const AgentsScreen({super.key, required this.api});
  final ApiService api;

  @override
  State<AgentsScreen> createState() => _AgentsScreenState();
}

class _AgentsScreenState extends State<AgentsScreen> {
  static const List<String> _decisions = [
    'Acknowledge evidence',
    'Request engineering review',
    'Escalate to reliability lead',
  ];

  final _assetController = TextEditingController(text: 'WTG-001');
  final _questionController = TextEditingController();
  final _hoursController = TextEditingController(text: '24');

  String _decision = _decisions[0];
  bool _asking = false;
  bool _recording = false;
  bool _simulating = false;

  Map<String, dynamic>? _answer;
  List<Map<String, dynamic>> _trail = [];
  Map<String, dynamic>? _scenarios;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadTrail();
  }

  @override
  void dispose() {
    _assetController.dispose();
    _questionController.dispose();
    _hoursController.dispose();
    super.dispose();
  }

  String get _asset => _assetController.text.trim().isEmpty
      ? 'WTG-001'
      : _assetController.text.trim();

  Future<void> _loadTrail() async {
    try {
      final data = await widget.api.listReviews(assetId: _asset, limit: 8);
      if (!mounted) return;
      setState(() {
        _trail = ((data['reviews'] as List<dynamic>?) ?? [])
            .map((e) => Map<String, dynamic>.from(e as Map))
            .toList();
      });
    } catch (_) {
      // Trail is best-effort; keep the screen usable offline.
    }
  }

  Future<void> _ask() async {
    final question = _questionController.text.trim();
    if (question.isEmpty) return;
    setState(() {
      _asking = true;
      _error = null;
    });
    try {
      final data = await widget.api.askAgents(assetId: _asset, question: question);
      if (!mounted) return;
      setState(() => _answer = data);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _asking = false);
    }
  }

  Future<void> _record() async {
    setState(() {
      _recording = true;
      _error = null;
    });
    try {
      await widget.api.recordReview(assetId: _asset, decision: _decision);
      await _loadTrail();
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _recording = false);
    }
  }

  Future<void> _runScenarios() async {
    setState(() {
      _simulating = true;
      _error = null;
    });
    try {
      final hours = double.tryParse(_hoursController.text.trim()) ?? 24.0;
      final data = await widget.api.runScenarios(assetId: _asset, hours: hours);
      if (!mounted) return;
      setState(() => _scenarios = data);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _simulating = false);
    }
  }

  Color _agentColor(String agent) {
    switch (agent) {
      case 'MIKA':
        return Colors.pinkAccent;
      case 'KAI':
        return Colors.cyanAccent;
      default:
        return Colors.amberAccent;
    }
  }

  @override
  Widget build(BuildContext context) {
    final scenarioRows = ((_scenarios?['scenarios'] as List<dynamic>?) ?? [])
        .map((e) => Map<String, dynamic>.from(e as Map))
        .toList();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('MIKA + KAI Agent Copilot',
              style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 4),
          const Text(
            'Ask the dual-agent council, record advisory-only human decisions, '
            'and compare parallel scenario futures. Decision-support only — '
            'no turbine commands.',
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: 320,
            child: TextField(
              controller: _assetController,
              decoration: const InputDecoration(
                labelText: 'Asset ID',
                border: OutlineInputBorder(),
              ),
              onSubmitted: (_) => _loadTrail(),
            ),
          ),
          const SizedBox(height: 16),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Ask the Agent Council',
                            style: Theme.of(context).textTheme.titleMedium),
                        const SizedBox(height: 8),
                        TextField(
                          controller: _questionController,
                          maxLines: 2,
                          decoration: const InputDecoration(
                            labelText:
                                'Why is bearing risk rising? When should engineering review it?',
                            border: OutlineInputBorder(),
                          ),
                        ),
                        const SizedBox(height: 8),
                        FilledButton.icon(
                          onPressed: _asking ? null : _ask,
                          icon: _asking
                              ? const SizedBox(
                                  width: 14,
                                  height: 14,
                                  child: CircularProgressIndicator(strokeWidth: 2))
                              : const Icon(Icons.forum_outlined),
                          label: const Text('Ask MIKA + KAI'),
                        ),
                        if (_error != null) ...[
                          const SizedBox(height: 8),
                          Text(_error!,
                              style: TextStyle(color: Colors.red.shade300)),
                        ],
                        if (_answer != null) ...[
                          const SizedBox(height: 12),
                          Text(
                            '${_answer!['agent']} · ${_answer!['risk_level']} risk · '
                            'review window ${(_answer!['review_window_days'] as num?)?.toStringAsFixed(1)} d',
                            style: TextStyle(
                              color: _agentColor('${_answer!['agent']}'),
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text('${_answer!['answer']}'),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 6,
                            runSpacing: 4,
                            children: ((_answer!['connected_sources']
                                    as List<dynamic>?) ??
                                [])
                                .map((s) => Chip(
                                      label: Text('$s',
                                          style: const TextStyle(fontSize: 11)),
                                      visualDensity: VisualDensity.compact,
                                    ))
                                .toList(),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Human decision gate',
                            style: Theme.of(context).textTheme.titleMedium),
                        const SizedBox(height: 8),
                        DropdownButtonFormField<String>(
                          value: _decision,
                          decoration: const InputDecoration(
                            labelText: 'Operator decision',
                            border: OutlineInputBorder(),
                          ),
                          items: _decisions
                              .map((d) =>
                                  DropdownMenuItem(value: d, child: Text(d)))
                              .toList(),
                          onChanged: (v) =>
                              setState(() => _decision = v ?? _decisions[0]),
                        ),
                        const SizedBox(height: 8),
                        FilledButton.icon(
                          onPressed: _recording ? null : _record,
                          icon: const Icon(Icons.fact_check_outlined),
                          label: const Text('Record Human Review'),
                        ),
                        const SizedBox(height: 10),
                        if (_trail.isEmpty)
                          const Text('No recorded reviews for this asset yet.')
                        else
                          Wrap(
                            spacing: 6,
                            runSpacing: 4,
                            children: _trail.map((row) {
                              return Chip(
                                label: Text(
                                  '#${row['id']} · ${row['decision']}',
                                  style: const TextStyle(fontSize: 11),
                                ),
                                visualDensity: VisualDensity.compact,
                              );
                            }).toList(),
                          ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Scenario lab · parallel futures',
                      style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      SizedBox(
                        width: 160,
                        child: TextField(
                          controller: _hoursController,
                          decoration: const InputDecoration(
                            labelText: 'Horizon (hours)',
                            border: OutlineInputBorder(),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      FilledButton.icon(
                        onPressed: _simulating ? null : _runScenarios,
                        icon: _simulating
                            ? const SizedBox(
                                width: 14,
                                height: 14,
                                child:
                                    CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.compare_arrows),
                        label: const Text('Run scenario comparison'),
                      ),
                      if (_scenarios != null) ...[
                        const SizedBox(width: 12),
                        Text(
                          'Best: ${_scenarios!['best_profile']} · '
                          'Worst: ${_scenarios!['worst_profile']}',
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 10),
                  if (scenarioRows.isNotEmpty)
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: DataTable(
                        columns: const [
                          DataColumn(label: Text('Profile')),
                          DataColumn(label: Text('Projected RUL (d)')),
                          DataColumn(label: Text('Wear Δ (%)')),
                          DataColumn(label: Text('Bearing L10 (h)')),
                          DataColumn(label: Text('Risk')),
                        ],
                        rows: scenarioRows.map((row) {
                          return DataRow(cells: [
                            DataCell(Text('${row['profile']}'
                                '${row['profile'] == _scenarios!['best_profile'] ? ' ★' : ''}')),
                            DataCell(Text(
                                '${(row['final_rul_days'] as num?)?.toStringAsFixed(1) ?? '—'}')),
                            DataCell(Text(
                                '${(row['wear_delta_pct'] as num?)?.toStringAsFixed(3) ?? '—'}')),
                            DataCell(Text(
                                '${(row['bearing_l10_hours'] as num?)?.toStringAsFixed(0) ?? '—'}')),
                            DataCell(Text('${row['risk_level'] ?? '—'}')),
                          ]);
                        }).toList(),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
