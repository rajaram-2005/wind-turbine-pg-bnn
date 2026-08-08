import 'package:flutter/material.dart';

import 'services/api_service.dart';
import 'screens/data_ingestion_screen.dart';
import 'screens/digital_twin_screen.dart';
import 'screens/fleet_reports_screen.dart';
import 'screens/aerozip_telemetry_screen.dart';
import 'screens/low_level_inference_screen.dart';
import 'screens/jobs_screen.dart';

void main() => runApp(const AeroVigilApp());

/// Root application widget. Provides a single shared [ApiService] instance and
/// a responsive navigation shell (rail on wide screens, bottom bar on mobile).
class AeroVigilApp extends StatelessWidget {
  const AeroVigilApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF0D9488);
    return MaterialApp(
      title: 'AeroVigilAI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: seed,
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF05121A),
        cardTheme: CardTheme(
          color: const Color(0xFF0C2029),
          elevation: 2,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
      ),
      home: const HomeShell(),
    );
  }
}

class _NavItem {
  const _NavItem(this.label, this.icon, this.builder);
  final String label;
  final IconData icon;
  final Widget Function(ApiService api) builder;
}

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});
  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  final ApiService _api = ApiService();
  int _index = 0;

  late final List<_NavItem> _items = [
    _NavItem('Digital Twin', Icons.speed, (api) => DigitalTwinScreen(api: api)),
    _NavItem('Ingestion', Icons.upload_file, (api) => DataIngestionScreen(api: api)),
    _NavItem('Fleet', Icons.table_chart, (api) => FleetReportsScreen(api: api)),
    _NavItem('AeroZip', Icons.compress, (api) => AeroZipTelemetryScreen(api: api)),
    _NavItem('Jobs', Icons.play_circle, (api) => JobsScreen(api: api)),
    _NavItem('Inference', Icons.code, (api) => LowLevelInferenceScreen(api: api)),
  ];

  @override
  Widget build(BuildContext context) {
    final isWide = MediaQuery.of(context).size.width >= 720;
    final body = _items[_index].builder(_api);

    if (isWide) {
      return Scaffold(
        body: Row(
          children: [
            NavigationRail(
              selectedIndex: _index,
              onDestinationSelected: (i) => setState(() => _index = i),
              labelType: NavigationRailLabelType.all,
              leading: const Padding(
                padding: EdgeInsets.symmetric(vertical: 16),
                child: Icon(Icons.wind_power, size: 30, color: Color(0xFF2DD4BF)),
              ),
              destinations: [
                for (final it in _items)
                  NavigationRailDestination(icon: Icon(it.icon), label: Text(it.label)),
              ],
            ),
            const VerticalDivider(width: 1),
            Expanded(child: SafeArea(child: body)),
          ],
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: Text('AeroVigilAI • ${_items[_index].label}')),
      body: SafeArea(child: body),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: [
          for (final it in _items)
            NavigationDestination(icon: Icon(it.icon), label: it.label),
        ],
      ),
    );
  }
}
