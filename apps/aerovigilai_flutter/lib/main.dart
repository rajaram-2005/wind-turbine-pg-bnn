import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

const _configuredApi = String.fromEnvironment('AEROVIGILAI_API_URL');

String defaultApiUrl() {
  if (_configuredApi.isNotEmpty) return _configuredApi;
  return defaultTargetPlatform == TargetPlatform.android
      ? 'http://10.0.2.2:8080'
      : 'http://127.0.0.1:8080';
}

void main() => runApp(const AeroVigilAIApp());

class AeroVigilAIApp extends StatelessWidget {
  const AeroVigilAIApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'AeroVigilAI',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff45d7e9), brightness: Brightness.dark),
          scaffoldBackgroundColor: const Color(0xff07111d),
          useMaterial3: true,
        ),
        home: AdvisoryConsole(apiUrl: defaultApiUrl()),
      );
}

class AdvisoryConsole extends StatefulWidget {
  const AdvisoryConsole({super.key, required this.apiUrl});
  final String apiUrl;
  @override
  State<AdvisoryConsole> createState() => _AdvisoryConsoleState();
}

class _AdvisoryConsoleState extends State<AdvisoryConsole> {
  final formKey = GlobalKey<FormState>();
  final asset = TextEditingController(text: 'WTG-042');
  final vibration = TextEditingController(text: '4.8');
  final temperature = TextEditingController(text: '82');
  final rpm = TextEditingController(text: '1780');
  final viscosity = TextEditingController(text: '12');
  final load = TextEditingController(text: '95');
  final rul = TextEditingController(text: '14.2');
  final epistemic = TextEditingController(text: '0.04');
  final aleatoric = TextEditingController(text: '0.12');
  final wind = TextEditingController();
  final power = TextEditingController();
  Map<String, dynamic>? result;
  String? error;
  bool loading = false;

  double n(TextEditingController c) => double.parse(c.text.trim());
  Future<void> assess() async {
    if (!formKey.currentState!.validate()) return;
    setState(() { loading = true; error = null; });
    final context = <String, dynamic>{};
    if (wind.text.trim().isNotEmpty) context['wind_speed_ms'] = n(wind);
    if (power.text.trim().isNotEmpty) context['power_output_kw'] = n(power);
    final payload = <String, dynamic>{
      'asset_id': asset.text.trim(),
      'telemetry': {'vibration_mms': n(vibration), 'temperature_c': n(temperature), 'rpm': n(rpm), 'oil_viscosity_cst': n(viscosity), 'load_pct': n(load)},
      'bnn_state': {'predicted_rul_days': n(rul), 'epistemic_uncertainty': n(epistemic), 'aleatoric_uncertainty': n(aleatoric)},
      if (context.isNotEmpty) 'physics_guided_context': context,
    };
    try {
      final response = await http.post(Uri.parse('${widget.apiUrl}/api/advisory'), headers: {'Content-Type': 'application/json'}, body: jsonEncode(payload));
      final decoded = jsonDecode(response.body);
      if (response.statusCode >= 300) throw Exception(decoded['detail'] ?? 'HTTP ${response.statusCode}');
      setState(() => result = Map<String, dynamic>.from(decoded));
    } catch (e) {
      setState(() => error = 'Unable to reach AeroVigilAI at ${widget.apiUrl}. $e');
    } finally { if (mounted) setState(() => loading = false); }
  }

  @override
  void dispose() { for (final c in [asset,vibration,temperature,rpm,viscosity,load,rul,epistemic,aleatoric,wind,power]) { c.dispose(); } super.dispose(); }
  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('AeroVigilAI'), actions: [Padding(padding: const EdgeInsets.all(12), child: Chip(label: const Text('ADVISORY ONLY'), backgroundColor: Colors.cyan.withOpacity(.15))) ]),
    body: LayoutBuilder(builder: (context, constraints) {
      final desktop = constraints.maxWidth >= 860;
      final input = _InputCard(formKey: formKey, fields: this, onAssess: assess, loading: loading);
      final output = _ResultCard(result: result, error: error, apiUrl: widget.apiUrl);
      return SingleChildScrollView(padding: const EdgeInsets.all(20), child: Center(child: ConstrainedBox(maxWidth: 1200, child: desktop ? Row(crossAxisAlignment: CrossAxisAlignment.start, children: [Expanded(child: input), const SizedBox(width: 20), Expanded(flex: 12, child: output)]) : Column(children: [input, const SizedBox(height: 16), output]))));
    }),
  );
}

class _InputCard extends StatelessWidget {
  const _InputCard({required this.formKey, required this.fields, required this.onAssess, required this.loading});
  final GlobalKey<FormState> formKey; final _AdvisoryConsoleState fields; final VoidCallback onAssess; final bool loading;
  Widget field(String label, TextEditingController c) => TextFormField(controller: c, keyboardType: TextInputType.number, validator: (v) => v == null || double.tryParse(v) == null ? 'Enter a number' : null, decoration: InputDecoration(labelText: label));
  @override Widget build(BuildContext context) => Card(child: Padding(padding: const EdgeInsets.all(20), child: Form(key: formKey, child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [const Text('Assess an asset', style: TextStyle(fontSize: 22,fontWeight: FontWeight.bold)), const SizedBox(height: 6), const Text('Local SCADA snapshot and optional physics context.'), const SizedBox(height: 18), TextFormField(controller: fields.asset, validator: (v) => v!.isEmpty ? 'Asset ID is required' : null, decoration: const InputDecoration(labelText: 'Asset ID')), field('Vibration (mm/s)',fields.vibration), field('Temperature (°C)',fields.temperature), field('High-speed RPM',fields.rpm), field('Oil viscosity (cSt)',fields.viscosity), field('Load (%)',fields.load), ExpansionTile(title: const Text('RUL model state'), children: [field('Predicted RUL (days)',fields.rul),field('Epistemic uncertainty',fields.epistemic),field('Aleatoric uncertainty',fields.aleatoric)]), ExpansionTile(title: const Text('Measured wind and power (optional)'), children: [TextField(controller: fields.wind, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Wind speed (m/s)')),TextField(controller: fields.power, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Power (kW)'))]), const SizedBox(height: 14), FilledButton(onPressed: loading ? null : onAssess, child: Padding(padding: const EdgeInsets.all(12), child: Text(loading ? 'Assessing…' : 'Run advisory')))])));
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result, required this.error, required this.apiUrl}); final Map<String,dynamic>? result; final String? error; final String apiUrl;
  @override Widget build(BuildContext context) { if (error != null) return Card(color: Colors.red.withOpacity(.14), child: Padding(padding: const EdgeInsets.all(20), child: Text(error!))); if (result == null) return Card(child: Padding(padding: const EdgeInsets.all(28), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [const Text('Ready for evidence',style: TextStyle(fontSize:22,fontWeight:FontWeight.bold)), const SizedBox(height:8), const Text('Submit a snapshot to receive the connected advisory.'), const SizedBox(height:20), Text('API: $apiUrl',style: const TextStyle(color: Colors.cyan))]))); final team = Map<String,dynamic>.from(result!['agent_team'] ?? {}); final physics = result!['physics_guided']; return Card(child: Padding(padding: const EdgeInsets.all(22), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(result!['asset_id'],style: const TextStyle(fontSize:26,fontWeight:FontWeight.bold)), const SizedBox(height:5), Chip(label: Text(team['risk_level'] ?? 'REVIEW')), const Divider(), Wrap(spacing:20,runSpacing:12,children:[_metric('Predicted RUL','${(result!['predicted_rul_days'] as num).toStringAsFixed(1)} days'),_metric('Inspection window','${(result!['suggested_inspection_window_days'] as num).toStringAsFixed(1)} days'),_metric('Epistemic σ','${(result!['epistemic_std'] as num).toStringAsFixed(3)}')]), const SizedBox(height:22), const Text('Connected evidence',style:TextStyle(fontWeight:FontWeight.bold)), Text(team['shared_summary'] ?? ''), if (physics != null) ...[const SizedBox(height:20),const Text('Physics-guided posterior',style:TextStyle(fontWeight:FontWeight.bold)),Text('${physics['target_name']}: ${(physics['target_mean'] as num).toStringAsFixed(3)} ± ${(physics['total_std'] as num).toStringAsFixed(3)}')], const SizedBox(height:20),const Text('Rationale',style:TextStyle(fontWeight:FontWeight.bold)),Text(result!['rationale'] ?? ''),const SizedBox(height:20),const Text('Decision support only. Qualified human review is required.',style:TextStyle(fontSize:12,color:Colors.grey))]))); }
  Widget _metric(String label,String value)=>Column(crossAxisAlignment:CrossAxisAlignment.start,children:[Text(label,style:const TextStyle(fontSize:12,color:Colors.grey)),Text(value,style:const TextStyle(fontSize:18,fontWeight:FontWeight.bold))]);
}
