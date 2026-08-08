import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

import '../services/api_service.dart';

/// Live digital-twin dashboard: gauges + synchronized time-series charts
/// + advisory + history table. Implements the gauge/chart UX requirement.
class DigitalTwinScreen extends StatefulWidget {
  const DigitalTwinScreen({super.key, required this.api});
  final ApiService api;

  @override
  State<DigitalTwinScreen> createState() => _DigitalTwinScreenState();
}

class _DigitalTwinScreenState extends State<DigitalTwinScreen> {
  Timer? _timer;
  final _rng = Random();
  final _assetController = TextEditingController(text: 'WTG-001');

  // Live values
  double _rpm = 1450;
  double _gearboxTemp = 68;
  double _vibration = 4.2;
  double _load = 75;
  double _oilVisc = 32;
  double? _rul;
  double? _epistemic;
  double? _aleatoric;
  String _source = 'simulated';
  String _advisorySource = '-';
  String _status = '-';

  final List<FlSpot> _rpmSeries = [];
  final List<FlSpot> _tempSeries = [];
  final List<FlSpot> _vibSeries = [];
  final List<FlSpot> _loadSeries = [];
  double _t = 0;

  List<Map<String, dynamic>> _history = [];
  List<Map<String, dynamic>> _durableHistory = [];

  @override
  void initState() {
    super.initState();
    _tick();
    _loadDurableHistory();
    _timer = Timer.periodic(const Duration(seconds: 2), (_) => _tick());
  }

  Future<void> _loadDurableHistory() async {
    try {
      final data = await widget.api.getTwinHistory(assetId: _assetController.text.trim(), limit: 20);
      final hist = (data['history'] as List?) ?? [];
      if (!mounted) return;
      setState(() {
        _durableHistory = hist.whereType<Map>().map((e) => Map<String, dynamic>.from(e as Map)).toList();
      });
    } catch (_) {}
  }

  @override
  void dispose() {
    _timer?.cancel();
    _assetController.dispose();
    super.dispose();
  }

  double? _asDouble(dynamic v) => v is num ? v.toDouble() : double.tryParse('$v');

  Future<void> _tick() async {
    try {
      final data = await widget.api.getDigitalTwinState(assetId: _assetController.text.trim());
      // data from /api/twin/status: contains last_state etc
      final last = data['last_state'] as Map<String, dynamic>?;
      final adv = (last?['advisory'] as Map<String, dynamic>?) ?? {};
      // Extract telemetry from last_state or fallback
      double rpm = _asDouble(last?['rpm']) ?? _asDouble(data['rpm']) ?? _asDouble(last?['telemetry']?['rpm']) ?? _rpm;
      double temp = _asDouble(last?['temperature_c']) ?? _asDouble(last?['gearbox_temp']) ?? _asDouble(data['temperature_c']) ?? _gearboxTemp;
      double vib = _asDouble(last?['vibration_mms']) ?? _vibration;
      double load = _asDouble(last?['load_pct']) ?? _load;
      double visc = _asDouble(last?['oil_viscosity_cst']) ?? _oilVisc;

      _rpm = rpm;
      _gearboxTemp = temp;
      _vibration = vib;
      _load = load;
      _oilVisc = visc;
      _rul = _asDouble(adv['predicted_rul_days']);
      _epistemic = _asDouble(adv['epistemic_std']);
      _aleatoric = _asDouble(adv['aleatoric_std']);
      _advisorySource = '${last?['advisory_source'] ?? data['advisory_source'] ?? 'model'}';
      _status = '${adv['status'] ?? ''}';
      _source = 'live: ${data['model_name'] ?? ''}';

      // append to history view (latest 5)
      final recTimestamp = last?['timestamp'] ?? '';
      _history.insert(0, {
        'ts': recTimestamp,
        'rpm': rpm,
        'temp': temp,
        'vib': vib,
        'load': load,
        'rul': _rul,
      });
      if (_history.length > 30) _history = _history.sublist(0, 30);
    } catch (_) {
      _rpm = (_rpm + _rng.nextDouble() * 60 - 30).clamp(900, 1800);
      _gearboxTemp = (_gearboxTemp + _rng.nextDouble() * 3 - 1.5).clamp(40, 95);
      _vibration = (_vibration + _rng.nextDouble() * 0.8 - 0.4).clamp(0.2, 12);
      _load = (_load + _rng.nextDouble() * 6 - 3).clamp(5, 100);
      _source = 'simulated';
      _advisorySource = 'simulated';
    }
    if (!mounted) return;
    setState(() {
      _t += 1;
      _rpmSeries.add(FlSpot(_t, _rpm));
      _tempSeries.add(FlSpot(_t, _gearboxTemp));
      _vibSeries.add(FlSpot(_t, _vibration));
      _loadSeries.add(FlSpot(_t, _load));
      if (_rpmSeries.length > 60) _rpmSeries.removeAt(0);
      if (_tempSeries.length > 60) _tempSeries.removeAt(0);
      if (_vibSeries.length > 60) _vibSeries.removeAt(0);
      if (_loadSeries.length > 60) _loadSeries.removeAt(0);
    });
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('Digital Twin', style: Theme.of(context).textTheme.headlineSmall),
              const SizedBox(width: 12),
              Chip(label: Text(_source), visualDensity: VisualDensity.compact),
              const Spacer(),
              SizedBox(
                width: 160,
                child: TextField(
                  controller: _assetController,
                  decoration: const InputDecoration(labelText: 'Asset ID', border: OutlineInputBorder(), isDense: true),
                  onSubmitted: (_) {
                    _tick();
                    _loadDurableHistory();
                  },
                ),
              ),
              const SizedBox(width: 8),
              IconButton(onPressed: () { _tick(); _loadDurableHistory(); }, icon: const Icon(Icons.refresh)),
            ],
          ),
          const SizedBox(height: 8),
          Text('Twin asset ${_assetController.text} • advisory source $_advisorySource • $_status',
              style: const TextStyle(color: Colors.white60, fontSize: 12)),
          const SizedBox(height: 20),

          // Advisory cards
          Wrap(
            spacing: 16,
            runSpacing: 16,
            children: [
              _InfoCard(label: 'Predicted RUL', value: _rul != null ? '${_rul!.toStringAsFixed(1)} days' : '—', icon: Icons.timelapse, color: const Color(0xFF2DD4BF)),
              _InfoCard(label: 'Epistemic σ', value: _epistemic != null ? '${_epistemic!.toStringAsFixed(2)}' : '—', icon: Icons.psychology, color: const Color(0xFF60A5FA)),
              _InfoCard(label: 'Aleatoric σ', value: _aleatoric != null ? '${_aleatoric!.toStringAsFixed(2)}' : '—', icon: Icons.grain, color: const Color(0xFFF59E0B)),
              _InfoCard(label: 'Cumulative Wear', value: '${(_vibration * 0.12).toStringAsFixed(3)}', icon: Icons.settings, color: const Color(0xFFEF4444)),
            ],
          ),
          const SizedBox(height: 20),

          // Gauges
          Wrap(
            spacing: 16,
            runSpacing: 16,
            children: [
              _GaugeCard(label: 'Generator RPM', value: _rpm, min: 900, max: 1800, unit: 'rpm'),
              _GaugeCard(label: 'Gearbox Temp', value: _gearboxTemp, min: 40, max: 100, unit: '°C'),
              _GaugeCard(label: 'Vibration', value: _vibration, min: 0, max: 12, unit: 'mm/s'),
              _GaugeCard(label: 'Load', value: _load, min: 0, max: 100, unit: '%'),
              _GaugeCard(label: 'Oil Viscosity', value: _oilVisc, min: 15, max: 60, unit: 'cSt'),
            ],
          ),
          const SizedBox(height: 24),

          // Charts
          Wrap(
            spacing: 16,
            runSpacing: 16,
            children: [
              SizedBox(width: 520, child: _ChartCard(title: 'Generator RPM (live)', spots: _rpmSeries, color: const Color(0xFF2DD4BF))),
              SizedBox(width: 520, child: _ChartCard(title: 'Gearbox Temperature (live)', spots: _tempSeries, color: const Color(0xFFF59E0B))),
              SizedBox(width: 520, child: _ChartCard(title: 'Vibration RMS (live)', spots: _vibSeries, color: const Color(0xFFF43F5E))),
              SizedBox(width: 520, child: _ChartCard(title: 'Load % (live)', spots: _loadSeries, color: const Color(0xFF60A5FA))),
            ],
          ),

          const SizedBox(height: 24),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Recent Twin History (in-memory buffer)', style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 12),
                  SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: DataTable(
                      columns: const [
                        DataColumn(label: Text('TS')),
                        DataColumn(label: Text('RPM')),
                        DataColumn(label: Text('Temp °C')),
                        DataColumn(label: Text('Vib mm/s')),
                        DataColumn(label: Text('Load %')),
                        DataColumn(label: Text('RUL d')),
                      ],
                      rows: [
                        for (final h in _history.take(10))
                          DataRow(cells: [
                            DataCell(Text('${h['ts']}'.toString().length >= 19 ? '${h['ts']}'.toString().substring(0, 19) : '${h['ts']}')),
                            DataCell(Text('${(h['rpm'] as num).toStringAsFixed(0)}')),
                            DataCell(Text('${(h['temp'] as num).toStringAsFixed(1)}')),
                            DataCell(Text('${(h['vib'] as num).toStringAsFixed(2)}')),
                            DataCell(Text('${(h['load'] as num).toStringAsFixed(1)}')),
                            DataCell(Text('${(h['rul'] as num?)?.toStringAsFixed(1) ?? '-'}')),
                          ]),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text('Durable Twin History (SQLite – /api/twin/history)', style: Theme.of(context).textTheme.titleSmall),
                      const Spacer(),
                      Chip(label: Text('${_durableHistory.length} rows')),
                    ],
                  ),
                  const SizedBox(height: 12),
                  if (_durableHistory.isEmpty)
                    const Text('No durable twin snapshots yet – hardware streams will populate this via POST /api/hardware/stream → record_twin_state().', style: TextStyle(color: Colors.white54, fontSize: 12))
                  else
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: DataTable(
                        columns: const [
                          DataColumn(label: Text('TS')),
                          DataColumn(label: Text('Source')),
                          DataColumn(label: Text('RUL (d)')),
                          DataColumn(label: Text('Wear')),
                          DataColumn(label: Text('Advisory')),
                        ],
                        rows: [
                          for (final h in _durableHistory.take(15))
                            DataRow(cells: [
                              DataCell(Text('${h['timestamp'] ?? h['ts'] ?? '-'}'.toString().length >= 19 ? '${h['timestamp'] ?? h['ts']}'.toString().substring(0, 19) : '${h['timestamp'] ?? h['ts']}' )),
                              DataCell(Text('${h['advisory_source'] ?? '-'}')),
                              DataCell(Text('${(h['advisory']?['predicted_rul_days'] as num?)?.toStringAsFixed(1) ?? '-'}')),
                              DataCell(Text('${(h['cumulative_wear'] as num?)?.toStringAsFixed(3) ?? '-'}')),
                              DataCell(Text('${h['advisory']?['status'] ?? h['advisory']?['risk_level'] ?? '-'}')),
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
    );
  }
}

class _InfoCard extends StatelessWidget {
  const _InfoCard({required this.label, required this.value, required this.icon, required this.color});
  final String label;
  final String value;
  final IconData icon;
  final Color color;
  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 220,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Icon(icon, color: color),
            const SizedBox(height: 8),
            Text(label, style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 4),
            Text(value, style: Theme.of(context).textTheme.titleMedium),
          ]),
        ),
      ),
    );
  }
}

class _GaugeCard extends StatelessWidget {
  const _GaugeCard({required this.label, required this.value, required this.min, required this.max, required this.unit});
  final String label;
  final double value;
  final double min;
  final double max;
  final String unit;

  @override
  Widget build(BuildContext context) {
    final frac = ((value - min) / (max - min)).clamp(0.0, 1.0);
    return SizedBox(
      width: 220,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Text(label, style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 16),
              SizedBox(
                height: 120,
                width: 120,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      height: 120,
                      width: 120,
                      child: CircularProgressIndicator(
                        value: frac,
                        strokeWidth: 10,
                        backgroundColor: Colors.white10,
                        valueColor: AlwaysStoppedAnimation(
                          Color.lerp(const Color(0xFF2DD4BF), const Color(0xFFEF4444), frac)!,
                        ),
                      ),
                    ),
                    Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(value.toStringAsFixed(1), style: Theme.of(context).textTheme.headlineSmall),
                        Text(unit, style: Theme.of(context).textTheme.bodySmall),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ChartCard extends StatelessWidget {
  const _ChartCard({required this.title, required this.spots, required this.color});
  final String title;
  final List<FlSpot> spots;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: spots.isEmpty
                  ? const Center(child: CircularProgressIndicator())
                  : LineChart(
                      LineChartData(
                        titlesData: const FlTitlesData(show: false),
                        gridData: const FlGridData(show: true, drawVerticalLine: false),
                        borderData: FlBorderData(show: false),
                        lineBarsData: [
                          LineChartBarData(
                            spots: spots,
                            isCurved: true,
                            color: color,
                            barWidth: 2.5,
                            dotData: const FlDotData(show: false),
                            belowBarData: BarAreaData(show: true, color: color.withOpacity(0.12)),
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
