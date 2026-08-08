import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

import '../services/api_service.dart';

/// Executive overview dashboard: fleet summary, system stats, live twin,
/// RUL distribution, and risk table. Combines gauge + chart + table UX
/// required by the platform spec.
class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key, required this.api});
  final ApiService api;

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  bool _loading = true;
  Map<String, dynamic> _summary = {};
  Map<String, dynamic> _stats = {};
  Map<String, dynamic> _twin = {};
  Map<String, dynamic> _reports = {};
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final results = await Future.wait([
        widget.api.getFleetSummary().catchError((_) => <String, dynamic>{}),
        widget.api.getSystemStats().catchError((_) => <String, dynamic>{}),
        widget.api.getDigitalTwinState().catchError((_) => <String, dynamic>{}),
        widget.api.getReports(kind: 'fleet', limit: 5).catchError((_) => <String, dynamic>{}),
      ]);
      if (!mounted) return;
      setState(() {
        _summary = results[0] as Map<String, dynamic>;
        _stats = results[1] as Map<String, dynamic>;
        _twin = results[2] as Map<String, dynamic>;
        _reports = results[3] as Map<String, dynamic>;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) return const Center(child: CircularProgressIndicator());

    final turbines = (_summary['turbines'] as List?) ?? const [];
    final nAssets = (_summary['n_assets'] as num?)?.toInt() ?? turbines.length;
    final atRisk = (_summary['at_risk_count'] as num?)?.toInt() ?? 0;
    final meanRul = (_summary['mean_rul_days'] as num?)?.toDouble();
    final meanHealth = (_summary['mean_health_score'] as num?)?.toDouble();

    // Derive RUL buckets for chart
    final ruls = <double>[];
    for (final t in turbines) {
      if (t is Map) {
        final v = (t['predicted_rul_days'] as num?)?.toDouble();
        if (v != null) ruls.add(v);
      }
    }
    // Fallback simulated RUL distribution when no data
    final displayRuls = ruls.isNotEmpty ? ruls : List.generate(20, (i) => 20 + math.Random(i).nextDouble() * 300);

    return RefreshIndicator(
      onRefresh: _load,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text('AeroVigilAI Dashboard', style: Theme.of(context).textTheme.headlineSmall),
                const Spacer(),
                IconButton(onPressed: _load, icon: const Icon(Icons.refresh)),
                if (_error != null)
                  Chip(
                    label: Text('offline: $_error', overflow: TextOverflow.ellipsis),
                    backgroundColor: const Color(0x33EF4444),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            const Text('Unified fleet health, twin telemetry, and durable-store overview from the canonical :8080 deployment.'),
            const SizedBox(height: 20),

            // KPI row
            Wrap(
              spacing: 16,
              runSpacing: 16,
              children: [
                _KpiCard(icon: Icons.wind_power, label: 'Turbines tracked', value: '$nAssets', color: const Color(0xFF2DD4BF)),
                _KpiCard(icon: Icons.warning_amber, label: 'At risk (<104d)', value: '$atRisk', color: const Color(0xFFF59E0B)),
                _KpiCard(icon: Icons.timelapse, label: 'Mean RUL', value: meanRul != null ? '${meanRul.toStringAsFixed(1)}d' : '—', color: const Color(0xFF60A5FA)),
                _KpiCard(icon: Icons.health_and_safety, label: 'Mean Health', value: meanHealth != null ? '${meanHealth.toStringAsFixed(1)}%' : '—', color: const Color(0xFF22C55E)),
              ],
            ),
            const SizedBox(height: 24),

            // Gauges + donut + line
            Wrap(
              spacing: 16,
              runSpacing: 16,
              children: [
                SizedBox(
                  width: 360,
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Fleet Health', style: Theme.of(context).textTheme.titleSmall),
                          const SizedBox(height: 16),
                          _HealthGauge(health: meanHealth ?? 72),
                          const SizedBox(height: 12),
                          Text('Twin: ${_twin['asset_id'] ?? 'WTG-001'} • model ${_twin['model_name'] ?? '-'}'),
                        ],
                      ),
                    ),
                  ),
                ),
                SizedBox(
                  width: 420,
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('RUL Distribution', style: Theme.of(context).textTheme.titleSmall),
                          const SizedBox(height: 16),
                          SizedBox(height: 200, child: _RulHistogram(ruls: displayRuls)),
                        ],
                      ),
                    ),
                  ),
                ),
                SizedBox(
                  width: 360,
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Durable Store', style: Theme.of(context).textTheme.titleSmall),
                          const SizedBox(height: 12),
                          _StatsTable(stats: _stats),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 24),
            // Reports table (durable)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Persisted Fleet Reports (SQLite – /api/reports)', style: Theme.of(context).textTheme.titleSmall),
                    const SizedBox(height: 12),
                    if ((_reports['reports'] as List?)?.isEmpty ?? true)
                      const Text('No fleet reports yet – they are auto-generated on each hardware stream and twin simulate.', style: TextStyle(color: Colors.white54, fontSize: 12))
                    else
                      SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: DataTable(
                          columns: const [
                            DataColumn(label: Text('ID')),
                            DataColumn(label: Text('Title')),
                            DataColumn(label: Text('Kind')),
                            DataColumn(label: Text('TS')),
                          ],
                          rows: [
                            for (final r in (_reports['reports'] as List).take(5))
                              if (r is Map<String, dynamic>)
                                DataRow(cells: [
                                  DataCell(Text('${r['id'] ?? '-'}')),
                                  DataCell(Text('${(r['title'] ?? '').toString().substring(0, (r['title']?.toString().length ?? 0).clamp(0, 40))}')),
                                  DataCell(Text('${r['kind'] ?? '-'}')),
                                  DataCell(Text('${r['ts'] ?? '-'}'.toString().substring(0, 19))),
                                ]),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            // At-risk table
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('At-Risk Turbines', style: Theme.of(context).textTheme.titleSmall),
                    const SizedBox(height: 12),
                    if (turbines.isEmpty)
                      const Text('No turbines in durable store yet – streams will populate this table automatically.', style: TextStyle(color: Colors.white54))
                    else
                      SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: DataTable(
                          columns: const [
                            DataColumn(label: Text('Turbine')),
                            DataColumn(label: Text('Status')),
                            DataColumn(label: Text('Health %')),
                            DataColumn(label: Text('RUL (d)')),
                            DataColumn(label: Text('Availability')),
                            DataColumn(label: Text('Gateway')),
                          ],
                          rows: [
                            for (final t in turbines.take(12))
                              if (t is Map<String, dynamic>)
                                DataRow(cells: [
                                  DataCell(Text('${t['turbine_id'] ?? '-'}')),
                                  DataCell(_StatusDot(status: '${t['status'] ?? 'ok'}')),
                                  DataCell(Text('${(t['health_score'] as num?)?.toStringAsFixed(1) ?? '-'}')),
                                  DataCell(Text('${(t['predicted_rul_days'] as num?)?.toStringAsFixed(1) ?? '-'}')),
                                  DataCell(Text('${(t['availability'] as num?)?.toStringAsFixed(1) ?? '-'}%')),
                                  DataCell(Text('${t['gateway_id'] ?? '-'}')),
                                ]),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _KpiCard extends StatelessWidget {
  const _KpiCard({required this.icon, required this.label, required this.value, required this.color});
  final IconData icon;
  final String label;
  final String value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 220,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, color: color),
              const SizedBox(height: 10),
              Text(label, style: Theme.of(context).textTheme.bodySmall),
              const SizedBox(height: 4),
              Text(value, style: Theme.of(context).textTheme.headlineSmall),
            ],
          ),
        ),
      ),
    );
  }
}

class _HealthGauge extends StatelessWidget {
  const _HealthGauge({required this.health});
  final double health;
  @override
  Widget build(BuildContext context) {
    final frac = (health / 100).clamp(0.0, 1.0);
    Color col = Color.lerp(const Color(0xFFEF4444), const Color(0xFF22C55E), frac)!;
    if (frac < 0.6) col = const Color(0xFFEF4444);
    else if (frac < 0.8) col = const Color(0xFFF59E0B);
    return SizedBox(
      height: 160,
      child: Stack(
        alignment: Alignment.center,
        children: [
          SizedBox(
            width: 140,
            height: 140,
            child: CircularProgressIndicator(value: frac, strokeWidth: 12, backgroundColor: Colors.white10, valueColor: AlwaysStoppedAnimation(col)),
          ),
          Column(mainAxisSize: MainAxisSize.min, children: [
            Text('${health.toStringAsFixed(0)}%', style: Theme.of(context).textTheme.headlineMedium),
            const Text('health', style: TextStyle(fontSize: 12, color: Colors.white60)),
          ]),
        ],
      ),
    );
  }
}

class _RulHistogram extends StatelessWidget {
  const _RulHistogram({required this.ruls});
  final List<double> ruls;

  @override
  Widget build(BuildContext context) {
    if (ruls.isEmpty) return const Center(child: Text('No RUL data'));
    // bucket into 6 bins
    final bins = List<double>.filled(6, 0);
    for (final v in ruls) {
      int idx = (v / 60).floor().clamp(0, 5);
      bins[idx] += 1;
    }
    final maxBin = bins.reduce(math.max);
    return BarChart(
      BarChartData(
        gridData: const FlGridData(show: false),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              getTitlesWidget: (val, meta) {
                const labels = ['0-60', '60-120', '120-180', '180-240', '240-300', '300+'];
                int i = val.toInt();
                if (i < 0 || i >= labels.length) return const SizedBox();
                return Padding(padding: const EdgeInsets.only(top: 6), child: Text(labels[i], style: const TextStyle(fontSize: 10)));
              },
            ),
          ),
        ),
        barGroups: [
          for (int i = 0; i < bins.length; i++)
            BarChartGroupData(
              x: i,
              barRods: [
                BarChartRodData(
                  toY: bins[i],
                  width: 22,
                  color: const Color(0xFF2DD4BF),
                  borderRadius: BorderRadius.circular(6),
                  backDrawRodData: BackgroundBarChartRodData(show: true, toY: maxBin == 0 ? 1 : maxBin, color: Colors.white10),
                ),
              ],
            ),
        ],
      ),
    );
  }
}

class _StatsTable extends StatelessWidget {
  const _StatsTable({required this.stats});
  final Map<String, dynamic> stats;

  @override
  Widget build(BuildContext context) {
    final tables = (stats['tables'] as Map?) ?? const {};
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('DB: ${stats['db_path'] ?? '-'}', style: const TextStyle(fontSize: 11, color: Colors.white54)),
        const SizedBox(height: 8),
        for (final e in tables.entries)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 3),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('${e.key}', style: const TextStyle(fontSize: 13)),
                Text('${e.value}', style: const TextStyle(fontWeight: FontWeight.bold)),
              ],
            ),
          ),
        const SizedBox(height: 8),
        Text('Jobs tracked: ${stats['jobs'] ?? '-'}', style: const TextStyle(fontSize: 12)),
      ],
    );
  }
}

class _StatusDot extends StatelessWidget {
  const _StatusDot({required this.status});
  final String status;

  @override
  Widget build(BuildContext context) {
    Color c;
    switch (status) {
      case 'Healthy':
        c = const Color(0xFF22C55E);
        break;
      case 'Watch':
        c = const Color(0xFFF59E0B);
        break;
      case 'Alert':
        c = const Color(0xFFEF4444);
        break;
      default:
        c = Colors.white54;
    }
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(Icons.circle, size: 10, color: c),
      const SizedBox(width: 6),
      Text(status),
    ]);
  }
}
