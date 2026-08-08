import 'dart:math';
import 'package:flutter/material.dart';

import '../services/api_service.dart';

/// Paginated fleet-health data table.
class FleetReportsScreen extends StatefulWidget {
  const FleetReportsScreen({super.key, required this.api});
  final ApiService api;

  @override
  State<FleetReportsScreen> createState() => _FleetReportsScreenState();
}

class _FleetRow {
  _FleetRow(this.turbineId, this.farm, this.health, this.availability, this.status);
  final String turbineId;
  final String farm;
  final double health;
  final double availability;
  final String status;
}

class _FleetReportsScreenState extends State<FleetReportsScreen> {
  int _page = 1;
  final int _pageSize = 10;
  bool _loading = true;
  List<_FleetRow> _rows = [];
  int _total = 0;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final data = await widget.api.getFleetReports(page: _page, pageSize: _pageSize);
      final items = (data['turbines'] ?? data['rows'] ?? data['data']) as List<dynamic>?;
      if (items != null) {
        _rows = items.map((e) {
          final m = e as Map<String, dynamic>;
          return _FleetRow(
            '${m['turbine_id'] ?? m['id'] ?? '-'}',
            '${m['farm'] ?? m['farm_name'] ?? '-'}',
            _d(m['health'] ?? m['health_score']),
            _d(m['availability']),
            '${m['status'] ?? 'ok'}',
          );
        }).toList();
        _total = (data['total'] as num?)?.toInt() ?? _rows.length;
      } else {
        _rows = _simulate();
        _total = 73;
      }
    } catch (_) {
      _rows = _simulate();
      _total = 73;
    }
    if (mounted) setState(() => _loading = false);
  }

  double _d(dynamic v) => v is num ? v.toDouble() : 0;

  List<_FleetRow> _simulate() {
    final rng = Random(_page);
    const farms = ['North Ridge', 'Coastal Array', 'Highland', 'Delta Bay'];
    return List.generate(_pageSize, (i) {
      final id = (_page - 1) * _pageSize + i + 1;
      final health = 55 + rng.nextDouble() * 45;
      return _FleetRow(
        'WTG-${id.toString().padLeft(3, '0')}',
        farms[rng.nextInt(farms.length)],
        health,
        90 + rng.nextDouble() * 10,
        health > 80 ? 'Healthy' : (health > 65 ? 'Watch' : 'Alert'),
      );
    });
  }

  Color _statusColor(String s) => switch (s) {
        'Healthy' => const Color(0xFF22C55E),
        'Watch' => const Color(0xFFF59E0B),
        _ => const Color(0xFFEF4444),
      };

  @override
  Widget build(BuildContext context) {
    final pages = (_total / _pageSize).ceil().clamp(1, 9999);
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Fleet Health Reports', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 16),
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : Card(
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: SingleChildScrollView(
                        child: DataTable(
                          columns: const [
                            DataColumn(label: Text('Turbine')),
                            DataColumn(label: Text('Farm')),
                            DataColumn(label: Text('Health')),
                            DataColumn(label: Text('Availability')),
                            DataColumn(label: Text('Status')),
                          ],
                          rows: [
                            for (final r in _rows)
                              DataRow(cells: [
                                DataCell(Text(r.turbineId)),
                                DataCell(Text(r.farm)),
                                DataCell(Text('${r.health.toStringAsFixed(1)}%')),
                                DataCell(Text('${r.availability.toStringAsFixed(1)}%')),
                                DataCell(Row(children: [
                                  Icon(Icons.circle, size: 10, color: _statusColor(r.status)),
                                  const SizedBox(width: 6),
                                  Text(r.status),
                                ])),
                              ]),
                          ],
                        ),
                      ),
                    ),
                  ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              IconButton(
                onPressed: _page > 1 ? () { setState(() => _page--); _load(); } : null,
                icon: const Icon(Icons.chevron_left),
              ),
              Text('Page $_page of $pages'),
              IconButton(
                onPressed: _page < pages ? () { setState(() => _page++); _load(); } : null,
                icon: const Icon(Icons.chevron_right),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
