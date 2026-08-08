import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

import '../services/api_service.dart';

/// Live digital-twin dashboard: gauges + synchronized time-series charts.
class DigitalTwinScreen extends StatefulWidget {
  const DigitalTwinScreen({super.key, required this.api});
  final ApiService api;

  @override
  State<DigitalTwinScreen> createState() => _DigitalTwinScreenState();
}

class _DigitalTwinScreenState extends State<DigitalTwinScreen> {
  Timer? _timer;
  final _rng = Random();
  double _rpm = 1450;
  double _gearboxTemp = 68;
  final List<FlSpot> _rpmSeries = [];
  final List<FlSpot> _tempSeries = [];
  double _t = 0;
  String _source = 'simulated';

  @override
  void initState() {
    super.initState();
    _tick();
    _timer = Timer.periodic(const Duration(seconds: 2), (_) => _tick());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _tick() async {
    // Try the live endpoint; gracefully fall back to a local simulation so the
    // dashboard is always populated during development.
    try {
      final data = await widget.api.getDigitalTwinState();
      final rpm = _asDouble(data['generator_rpm']) ?? _asDouble(data['rpm']);
      final temp = _asDouble(data['gearbox_temp']) ?? _asDouble(data['temperature']);
      if (rpm != null) _rpm = rpm;
      if (temp != null) _gearboxTemp = temp;
      _source = 'live';
    } catch (_) {
      _rpm = (_rpm + _rng.nextDouble() * 60 - 30).clamp(900, 1800);
      _gearboxTemp = (_gearboxTemp + _rng.nextDouble() * 3 - 1.5).clamp(40, 95);
      _source = 'simulated';
    }
    setState(() {
      _t += 1;
      _rpmSeries.add(FlSpot(_t, _rpm));
      _tempSeries.add(FlSpot(_t, _gearboxTemp));
      if (_rpmSeries.length > 40) _rpmSeries.removeAt(0);
      if (_tempSeries.length > 40) _tempSeries.removeAt(0);
    });
  }

  double? _asDouble(dynamic v) => v is num ? v.toDouble() : null;

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
            ],
          ),
          const SizedBox(height: 20),
          Wrap(
            spacing: 16,
            runSpacing: 16,
            children: [
              _GaugeCard(label: 'Generator RPM', value: _rpm, min: 900, max: 1800, unit: 'rpm'),
              _GaugeCard(label: 'Gearbox Temp', value: _gearboxTemp, min: 40, max: 100, unit: '°C'),
            ],
          ),
          const SizedBox(height: 24),
          _ChartCard(title: 'Generator RPM (live)', spots: _rpmSeries, color: const Color(0xFF2DD4BF)),
          const SizedBox(height: 16),
          _ChartCard(title: 'Gearbox Temperature (live)', spots: _tempSeries, color: const Color(0xFFF59E0B)),
        ],
      ),
    );
  }
}

class _GaugeCard extends StatelessWidget {
  const _GaugeCard({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.unit,
  });
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
                        Text(value.toStringAsFixed(0),
                            style: Theme.of(context).textTheme.headlineSmall),
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
                            belowBarData: BarAreaData(
                              show: true,
                              color: color.withOpacity(0.12),
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
