import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

/// Centralized API service for the AeroVigilAI native console.
///
/// Every network call in the app routes through this class. The base URL
/// points at the single canonical AeroVigil deployment on
/// `http://localhost:8080`.
class ApiService {
  ApiService({String? baseUrl}) : baseUrl = baseUrl ?? const String.fromEnvironment('API_BASE', defaultValue: 'http://localhost:8080');

  /// Canonical AeroVigil server. Override for device/emulator networking
  /// (e.g. `http://10.0.2.2:8080` on the Android emulator via --dart-define=API_BASE=http://10.0.2.2:8080).
  static const String defaultBaseUrl = String.fromEnvironment('API_BASE', defaultValue: 'http://localhost:8080');

  final String baseUrl;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Map<String, String> get _jsonHeaders => const {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };

  // ── system ────────────────────────────────────────────────────────────

  /// Root liveness and service discovery, including the connected MIKA + KAI
  /// agent mesh shared by the web console and native dashboards.
  Future<Map<String, dynamic>> getHealth() async {
    final resp = await http.get(_uri('/health'), headers: _jsonHeaders);
    return _handle(resp);
  }

  // ── ingestion ─────────────────────────────────────────────────────────

  /// Upload an offline USB/CSV file to the ingestion endpoint over HTTPS.
  Future<Map<String, dynamic>> uploadDataFile({
    required String filename,
    required Uint8List bytes,
    void Function(double progress)? onProgress,
  }) async {
    final request = http.MultipartRequest('POST', _uri('/api/telemetry/upload'))
      ..fields['source'] = 'usb'
      ..files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    onProgress?.call(0.1);
    final streamed = await request.send();
    onProgress?.call(0.9);
    final body = await streamed.stream.bytesToString();
    onProgress?.call(1.0);
    if (streamed.statusCode >= 200 && streamed.statusCode < 300) {
      return _safeJson(body, fallback: {'status': 'ok', 'filename': filename});
    }
    throw ApiException('Upload failed (${streamed.statusCode})', body);
  }

  /// Import a SCADA export by signed HTTPS cloud URL.
  Future<Map<String, dynamic>> importCloudUrl({
    required String url,
    void Function(double progress)? onProgress,
  }) async {
    onProgress?.call(0.3);
    final resp = await http.post(
      _uri('/api/telemetry/import'),
      headers: _jsonHeaders,
      body: jsonEncode({'url': url}),
    );
    onProgress?.call(1.0);
    return _handle(resp);
  }

  /// List recent offline imports (USB / cloud / API provenance).
  Future<Map<String, dynamic>> getImports({int limit = 25}) async {
    final resp = await http.get(_uri('/api/imports?limit=$limit'), headers: _jsonHeaders);
    return _handle(resp);
  }

  // ── twin ──────────────────────────────────────────────────────────────

  /// Fetch the live digital-twin state (RPM, temperatures, sim metrics).
  Future<Map<String, dynamic>> getDigitalTwinState({String assetId = 'WTG-001'}) async {
    final resp = await http.get(
      _uri('/api/twin/status?asset_id=$assetId'),
      headers: _jsonHeaders,
    );
    return _handle(resp);
  }

  /// Fleet report markdown (GET /api/fleet/report returns text/markdown).
  Future<String> getFleetReportMarkdown() async {
    final resp = await http.get(_uri('/api/fleet/report'), headers: {'Accept': 'text/markdown'});
    if (resp.statusCode >= 200 && resp.statusCode < 300) return resp.body;
    throw ApiException('Fleet report failed (${resp.statusCode})', resp.body);
  }

  // ── fleet ─────────────────────────────────────────────────────────────

  /// Fetch a page of aggregate fleet-health rows (legacy report endpoint).
  Future<Map<String, dynamic>> getFleetReports({int page = 1, int pageSize = 10}) async {
    final resp = await http.get(
      _uri('/api/fleet/report?page=$page&page_size=$pageSize'),
      headers: _jsonHeaders,
    );
    // This endpoint returns markdown when Accept is missing; try JSON fallback to summary
    try {
      final map = _safeJson(resp.body, fallback: {});
      if (map.containsKey('turbines') || map.containsKey('rows') || map.containsKey('data')) {
        return map;
      }
    } catch (_) {}
    // Fall back to durable summary which is always JSON
    return getFleetSummary();
  }

  /// Durable fleet summary from SQLite (GET /api/fleet/summary) – authoritative.
  Future<Map<String, dynamic>> getFleetSummary() async {
    final resp = await http.get(_uri('/api/fleet/summary'), headers: _jsonHeaders);
    return _handle(resp);
  }

  /// System stats: row counts + DB location (GET /api/system/stats).
  Future<Map<String, dynamic>> getSystemStats() async {
    final resp = await http.get(_uri('/api/system/stats'), headers: _jsonHeaders);
    return _handle(resp);
  }

  /// Durable twin-state history (GET /api/twin/history).
  Future<Map<String, dynamic>> getTwinHistory({String assetId = 'WTG-001', int limit = 30}) async {
    final resp = await http.get(_uri('/api/twin/history?asset_id=$assetId&limit=$limit'), headers: _jsonHeaders);
    return _handle(resp);
  }

  /// List persisted reports (GET /api/reports).
  Future<Map<String, dynamic>> getReports({String? kind, int limit = 20}) async {
    final qp = kind != null ? '?kind=$kind&limit=$limit' : '?limit=$limit';
    final resp = await http.get(_uri('/api/reports$qp'), headers: _jsonHeaders);
    return _handle(resp);
  }

  // ── hardware / AeroZip ────────────────────────────────────────────────

  /// Fetch AeroZip compression / bandwidth / restoration telemetry.
  /// Returns latest persisted readings for dashboards.
  Future<Map<String, dynamic>> getAeroZipTelemetry({int limit = 80}) async {
    final resp = await http.get(
      _uri('/api/hardware/latest?limit=$limit'),
      headers: _jsonHeaders,
    );
    return _handle(resp);
  }

  /// Alias: latest hardware readings for table views.
  Future<Map<String, dynamic>> getHardwareLatest({int limit = 100}) async {
    final resp = await http.get(
      _uri('/api/hardware/latest?limit=$limit'),
      headers: _jsonHeaders,
    );
    return _handle(resp);
  }

  // ── model ─────────────────────────────────────────────────────────────

  /// Submit a manual JSON payload directly to the canonical model endpoint.
  Future<Map<String, dynamic>> postModelInference(Map<String, dynamic> payload) async {
    final resp = await http.post(
      _uri('/api/model'),
      headers: _jsonHeaders,
      body: jsonEncode(payload),
    );
    return _handle(resp);
  }

  // ── jobs ──────────────────────────────────────────────────────────────

  /// Queue a framework job (train/evaluate/export/federated/active-learning/explain).
  Future<Map<String, dynamic>> queueJob(String jobType, {List<String>? args}) async {
    final resp = await http.post(
      _uri('/api/jobs/$jobType'),
      headers: _jsonHeaders,
      body: jsonEncode({'args': args ?? const <String>[]}),
    );
    return _handle(resp);
  }

  /// Poll a queued job's status and recent logs.
  Future<Map<String, dynamic>> getJobStatus(String jobId) async {
    final resp = await http.get(_uri('/api/jobs/$jobId'), headers: _jsonHeaders);
    return _handle(resp);
  }

  /// List recently queued jobs (newest first) for the jobs dashboard.
  Future<Map<String, dynamic>> listJobs({int limit = 25}) async {
    final resp = await http.get(_uri('/api/jobs?limit=$limit'), headers: _jsonHeaders);
    return _handle(resp);
  }

  // ── MIKA + KAI agent copilot ──────────────────────────────────────────

  /// Ask the MIKA + KAI agent council about an asset. Physics questions route
  /// to KAI, maintenance-planning questions to MIKA, everything else to the
  /// council consensus. Answers are rebuilt from live twin evidence.
  Future<Map<String, dynamic>> askAgents({
    required String assetId,
    required String question,
    String model = 'GE-1.5',
  }) async {
    final resp = await http.post(
      _uri('/api/agent/ask'),
      headers: _jsonHeaders,
      body: jsonEncode({'asset_id': assetId, 'model': model, 'question': question}),
    );
    return _handle(resp);
  }

  /// Record an advisory-only operator decision at the human decision gate.
  Future<Map<String, dynamic>> recordReview({
    required String assetId,
    required String decision,
    String? note,
  }) async {
    final resp = await http.post(
      _uri('/api/agent/review'),
      headers: _jsonHeaders,
      body: jsonEncode({'asset_id': assetId, 'decision': decision, 'note': note}),
    );
    return _handle(resp);
  }

  /// Read back the durable human-review audit trail for an asset.
  Future<Map<String, dynamic>> listReviews({String? assetId, int limit = 20}) async {
    final query = StringBuffer('/api/agent/reviews?limit=$limit');
    if (assetId != null && assetId.isNotEmpty) {
      query.write('&asset_id=${Uri.encodeQueryComponent(assetId)}');
    }
    final resp = await http.get(_uri(query.toString()), headers: _jsonHeaders);
    return _handle(resp);
  }

  /// Scenario Lab: run parallel operating-profile futures on a forked twin.
  Future<Map<String, dynamic>> runScenarios({
    required String assetId,
    String model = 'GE-1.5',
    double hours = 24.0,
  }) async {
    final resp = await http.post(
      _uri('/api/twin/scenarios'),
      headers: _jsonHeaders,
      body: jsonEncode({'asset_id': assetId, 'model': model, 'hours': hours}),
    );
    return _handle(resp);
  }

  Map<String, dynamic> _handle(http.Response resp) {
    if (resp.statusCode >= 200 && resp.statusCode < 300) {
      return _safeJson(resp.body, fallback: {});
    }
    throw ApiException('Request failed (${resp.statusCode})', resp.body);
  }

  Map<String, dynamic> _safeJson(String body, {required Map<String, dynamic> fallback}) {
    if (body.isEmpty) return fallback;
    final decoded = jsonDecode(body);
    if (decoded is Map<String, dynamic>) return decoded;
    return {'data': decoded};
  }
}

/// Thrown when an API call returns a non-2xx status.
class ApiException implements Exception {
  ApiException(this.message, [this.body]);
  final String message;
  final String? body;
  @override
  String toString() => 'ApiException: $message${body != null ? ' – $body' : ''}';
}
