import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

/// Centralized API service for the AeroVigilAI native console.
///
/// Every network call in the app routes through this class. The base URL
/// points at the single canonical AeroVigil deployment on
/// `http://localhost:8080`.
class ApiService {
  ApiService({String? baseUrl}) : baseUrl = baseUrl ?? defaultBaseUrl;

  /// Canonical AeroVigil server. Override for device/emulator networking
  /// (e.g. `http://10.0.2.2:8080` on the Android emulator).
  static const String defaultBaseUrl = 'http://localhost:8080';

  final String baseUrl;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Map<String, String> get _jsonHeaders => const {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };

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
  ///
  /// The server parses the signed URL, fetches the object server-side and
  /// records the import with `source=cloud`. URLs must be https (or
  /// localhost for development).
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

  /// Fetch the live digital-twin state (RPM, temperatures, sim metrics).
  Future<Map<String, dynamic>> getDigitalTwinState() async {
    final resp = await http.get(_uri('/api/twin/status'), headers: _jsonHeaders);
    return _handle(resp);
  }

  /// Fetch a page of aggregate fleet-health rows.
  Future<Map<String, dynamic>> getFleetReports({int page = 1, int pageSize = 10}) async {
    final resp = await http.get(
      _uri('/api/fleet/report?page=$page&page_size=$pageSize'),
      headers: _jsonHeaders,
    );
    return _handle(resp);
  }

  /// Fetch AeroZip compression / bandwidth / restoration telemetry.
  Future<Map<String, dynamic>> getAeroZipTelemetry() async {
    final resp = await http.get(_uri('/api/hardware/latest?limit=50'), headers: _jsonHeaders);
    return _handle(resp);
  }

  /// Submit a manual JSON payload directly to the canonical model endpoint.
  Future<Map<String, dynamic>> postModelInference(Map<String, dynamic> payload) async {
    final resp = await http.post(
      _uri('/api/model'),
      headers: _jsonHeaders,
      body: jsonEncode(payload),
    );
    return _handle(resp);
  }

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
