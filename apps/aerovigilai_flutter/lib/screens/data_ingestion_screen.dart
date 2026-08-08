import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

import '../services/api_service.dart';

/// Offline data ingestion: USB/CSV file upload and signed HTTPS cloud URL
/// import. Both record the import in the durable store with provenance.
/// Now also shows import log table + system stats.
class DataIngestionScreen extends StatefulWidget {
  const DataIngestionScreen({super.key, required this.api});
  final ApiService api;

  @override
  State<DataIngestionScreen> createState() => _DataIngestionScreenState();
}

class _DataIngestionScreenState extends State<DataIngestionScreen> {
  double? _progress;
  String? _status;
  String? _selectedName;

  final _urlController = TextEditingController(
    text: 'https://bucket.example.com/scada/wtg-042.csv?X-Amz-Signature=…',
  );
  String? _cloudStatus;
  bool _cloudError = false;
  bool _busy = false;

  List<Map<String, dynamic>> _imports = [];
  Map<String, dynamic> _sysStats = {};

  @override
  void initState() {
    super.initState();
    _loadMeta();
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _loadMeta() async {
    try {
      final imports = await widget.api.getImports(limit: 25);
      final stats = await widget.api.getSystemStats();
      if (!mounted) return;
      setState(() {
        _imports = ((imports['imports'] as List?) ?? []).whereType<Map>().map((e) => Map<String, dynamic>.from(e as Map)).toList();
        _sysStats = stats;
      });
    } catch (_) {}
  }

  Future<void> _pickAndUpload() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['csv', 'json', 'parquet', 'zip'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;
    final file = result.files.single;
    final bytes = file.bytes;
    if (bytes == null) {
      setState(() => _status = 'Could not read file bytes.');
      return;
    }
    setState(() {
      _selectedName = file.name;
      _progress = 0;
      _status = 'Uploading …';
    });
    try {
      final resp = await widget.api.uploadDataFile(
        filename: file.name,
        bytes: bytes,
        onProgress: (p) => setState(() => _progress = p),
      );
      setState(() => _status = 'Uploaded (import #${resp['import_id']}): ${resp['filename']} ${resp['bytes']} bytes source=${resp['source']}');
      await _loadMeta();
    } catch (e) {
      setState(() => _status = 'Failed: $e');
    } finally {
      setState(() => _progress = null);
    }
  }

  Future<void> _importCloudUrl() async {
    final url = _urlController.text.trim();
    if (url.isEmpty) return;
    setState(() {
      _busy = true;
      _cloudError = false;
      _cloudStatus = 'Importing from signed HTTPS URL …';
    });
    try {
      final resp = await widget.api.importCloudUrl(url: url);
      setState(() {
        _cloudStatus = 'Imported: ${resp['filename']} (${resp['bytes']} bytes, import #${resp['import_id']}, source ${resp['source']})';
      });
      await _loadMeta();
    } catch (e) {
      setState(() {
        _cloudError = true;
        _cloudStatus = 'Cloud import failed: $e';
      });
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Data Ingestion', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 8),
          const Text('Import SCADA exports from USB media, or point the server at a signed HTTPS cloud URL it can fetch and record. All imports are durable in SQLite.'),
          const SizedBox(height: 24),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('USB / local file', style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      const Icon(Icons.usb, color: Color(0xFF2DD4BF)),
                      const SizedBox(width: 12),
                      Expanded(child: Text(_selectedName ?? 'No file selected', overflow: TextOverflow.ellipsis)),
                      FilledButton.icon(
                        onPressed: _progress == null ? _pickAndUpload : null,
                        icon: const Icon(Icons.folder_open),
                        label: const Text('Choose & Upload'),
                      ),
                    ],
                  ),
                  if (_progress != null) ...[
                    const SizedBox(height: 16),
                    LinearProgressIndicator(value: _progress),
                  ],
                  if (_status != null) ...[
                    const SizedBox(height: 16),
                    SelectableText(_status!),
                  ],
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
                  Text('Signed cloud URL', style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 6),
                  const Text(
                    'Paste a signed HTTPS URL (pre-signed S3/GCS/Azure SAS). The server fetches server-side and records the import as source=cloud.',
                    style: TextStyle(fontSize: 13, color: Colors.white60),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _urlController,
                    decoration: const InputDecoration(labelText: 'Signed HTTPS object URL', border: OutlineInputBorder(), hintText: 'https://…'),
                    keyboardType: TextInputType.url,
                  ),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      FilledButton.icon(
                        onPressed: _busy ? null : _importCloudUrl,
                        icon: _busy ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.cloud_download),
                        label: const Text('Import from URL'),
                      ),
                      const SizedBox(width: 12),
                      if (_cloudStatus != null)
                        Expanded(
                          child: SelectableText(
                            _cloudStatus!,
                            style: TextStyle(color: _cloudError ? const Color(0xFFEF4444) : null, fontSize: 13),
                          ),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text('Recent Imports', style: Theme.of(context).textTheme.titleSmall),
                      const Spacer(),
                      IconButton(onPressed: _loadMeta, icon: const Icon(Icons.refresh)),
                    ],
                  ),
                  const SizedBox(height: 12),
                  if (_imports.isEmpty)
                    const Text('No imports yet – upload a file or import a signed URL.', style: TextStyle(color: Colors.white54))
                  else
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: DataTable(
                        columns: const [
                          DataColumn(label: Text('ID')),
                          DataColumn(label: Text('Filename')),
                          DataColumn(label: Text('Source')),
                          DataColumn(label: Text('Bytes')),
                          DataColumn(label: Text('Type')),
                        ],
                        rows: [
                          for (final r in _imports.take(15))
                            DataRow(cells: [
                              DataCell(Text('${r['id'] ?? '-'}')),
                              DataCell(Text('${r['filename'] ?? '-'}')),
                              DataCell(Chip(label: Text('${r['source'] ?? '-'}'), visualDensity: VisualDensity.compact)),
                              DataCell(Text('${r['size_bytes'] ?? '-'}')),
                              DataCell(Text('${r['content_type'] ?? '-'}')),
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
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('System Stats – durable store', style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 12),
                  SelectableText(_sysStats.isEmpty ? 'Loading…' : _sysStats.toString(), style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
