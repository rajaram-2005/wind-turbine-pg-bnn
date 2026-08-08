import 'dart:async';
import 'package:flutter/material.dart';

import '../services/api_service.dart';

/// Framework job dashboard: submit train / evaluate / export / federated /
/// active-learning / explain jobs and follow their status + live logs.
class JobsScreen extends StatefulWidget {
  const JobsScreen({super.key, required this.api});
  final ApiService api;

  @override
  State<JobsScreen> createState() => _JobsScreenState();
}

class _JobsScreenState extends State<JobsScreen> {
  static const List<({String id, String label})> _jobTypes = [
    (id: 'train', label: 'train — PG-BNN training'),
    (id: 'evaluate', label: 'evaluate — RMSE/NLL/calibration'),
    (id: 'export', label: 'export — ONNX edge model'),
    (id: 'federated', label: 'federated — fleet FedAvg'),
    (id: 'active-learning', label: 'active-learning — uncertainty sampling'),
    (id: 'explain', label: 'explain — physics SHAP report'),
  ];

  String _selectedType = 'train';
  final _argsController = TextEditingController();
  bool _queuing = false;

  List<Map<String, dynamic>> _jobs = [];
  Map<String, dynamic>? _detail;
  bool _loadingJobs = true;
  Timer? _poll;
  String? _pollingId;

  @override
  void initState() {
    super.initState();
    _loadJobs();
  }

  @override
  void dispose() {
    _poll?.cancel();
    _argsController.dispose();
    super.dispose();
  }

  Future<void> _loadJobs() async {
    try {
      final data = await widget.api.listJobs(limit: 15);
      final jobs = (data['jobs'] as List<dynamic>? ?? [])
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList();
      if (!mounted) return;
      setState(() {
        _jobs = jobs;
        _loadingJobs = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _loadingJobs = false);
    }
  }

  Future<void> _queue() async {
    setState(() => _queuing = true);
    try {
      final args = _argsController.text.trim().isEmpty
          ? const <String>[]
          : _argsController.text.trim().split(RegExp(r'\s+'));
      final resp = await widget.api.queueJob(_selectedType, args: args);
      _argsController.clear();
      await _loadJobs();
      _startPolling(resp['job_id'] as String?);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Job ${_selectedType} queued → ${resp['job_id']}')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Queue failed: $e'), backgroundColor: const Color(0xFFEF4444)),
        );
      }
    } finally {
      if (mounted) setState(() => _queuing = false);
    }
  }

  void _startPolling(String? jobId) {
    _poll?.cancel();
    if (jobId == null) return;
    setState(() {
      _pollingId = jobId;
      _detail = null;
    });
    _poll = Timer.periodic(const Duration(seconds: 2), (_) => _pollJob(jobId));
    _pollJob(jobId);
  }

  Future<void> _pollJob(String jobId) async {
    try {
      final data = await widget.api.getJobStatus(jobId);
      if (!mounted) return;
      setState(() => _detail = data);
      final status = data['status'];
      if (status == 'Completed' || status == 'Failed') {
        _poll?.cancel();
        await _loadJobs();
      }
    } catch (_) {
      _poll?.cancel();
    }
  }

  String _statusText(Map<String, dynamic>? j) => '${j?['status'] ?? '—'}';

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Framework Jobs', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 4),
          const Text('Queue and monitor physics-guided framework jobs (durable queue, '
              'live logs via GET /api/jobs/{id}).'),
          const SizedBox(height: 20),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Submit a job', style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 12),
                  DropdownButtonFormField<String>(
                    initialValue: _selectedType,
                    decoration: const InputDecoration(labelText: 'Job type'),
                    items: [
                      for (final t in _jobTypes)
                        DropdownMenuItem(value: t.id, child: Text(t.label)),
                    ],
                    onChanged: (v) => setState(() => _selectedType = v ?? _selectedType),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _argsController,
                    decoration: const InputDecoration(
                      labelText: 'Extra CLI args (optional)',
                      hintText: '--rounds 1 --clients 1',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 16),
                  FilledButton.icon(
                    onPressed: _queuing ? null : _queue,
                    icon: _queuing
                        ? const SizedBox(
                            width: 16, height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.play_arrow),
                    label: const Text('Queue job'),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 20),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text('Recent jobs', style: Theme.of(context).textTheme.titleSmall),
                      const Spacer(),
                      IconButton(
                        onPressed: _loadJobs,
                        icon: const Icon(Icons.refresh, size: 20),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (_loadingJobs)
                    const Center(child: Padding(
                      padding: EdgeInsets.all(16),
                      child: CircularProgressIndicator(),
                    ))
                  else if (_jobs.isEmpty)
                    const Padding(
                      padding: EdgeInsets.all(8),
                      child: Text('No jobs yet — queue one above.',
                          style: TextStyle(color: Colors.white54)),
                    )
                  else
                    for (final j in _jobs)
                      InkWell(
                        onTap: () => _startPolling(j['job_id'] as String?),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(vertical: 8),
                          child: Row(
                            children: [
                              _StatusPill(status: _statusText(j)),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  '${j['job_type']}  •  ${j['job_id']}',
                                  overflow: TextOverflow.ellipsis,
                                  style: const TextStyle(fontSize: 13),
                                ),
                              ),
                              Text(
                                _fmtTime(j['created_at']),
                                style: Theme.of(context).textTheme.bodySmall,
                              ),
                            ],
                          ),
                        ),
                      ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 20),
          if (_detail != null) _JobDetailCard(job: _detail!),
        ],
      ),
    );
  }

  String _fmtTime(dynamic ts) {
    if (ts is! num) return '';
    final dt = DateTime.fromMillisecondsSinceEpoch((ts * 1000).round());
    final local = dt.toLocal();
    String two(int v) => v.toString().padLeft(2, '0');
    return '${local.year}-${two(local.month)}-${two(local.day)} ${two(local.hour)}:${two(local.minute)}';
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.status});
  final String status;

  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      'Completed' => const Color(0xFF22C55E),
      'Running' => const Color(0xFF60A5FA),
      'Failed' => const Color(0xFFEF4444),
      _ => const Color(0xFFF59E0B),
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.5)),
      ),
      child: Text(status, style: TextStyle(fontSize: 11, color: color)),
    );
  }
}

class _JobDetailCard extends StatelessWidget {
  const _JobDetailCard({required this.job});
  final Map<String, dynamic> job;

  @override
  Widget build(BuildContext context) {
    final logs = (job['logs'] as List<dynamic>? ?? []).cast<String>();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                _StatusPill(status: '${job['status']}'),
                const SizedBox(width: 12),
                Expanded(
                  child: Text('Live log — ${job['job_type']}',
                      style: Theme.of(context).textTheme.titleSmall),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              constraints: const BoxConstraints(maxHeight: 320),
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFF06131B),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.white10),
              ),
              child: SingleChildScrollView(
                child: SelectableText(
                  logs.isEmpty ? '(no logs yet)' : logs.join('\n'),
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12, height: 1.5),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
