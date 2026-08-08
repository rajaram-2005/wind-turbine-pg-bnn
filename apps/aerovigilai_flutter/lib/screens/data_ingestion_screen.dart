import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

import '../services/api_service.dart';

/// Offline data ingestion: pick USB/CSV files and upload them over HTTPS.
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
      setState(() => _status = 'Uploaded: ${resp.toString()}');
    } catch (e) {
      setState(() => _status = 'Failed: $e');
    } finally {
      setState(() => _progress = null);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Offline Data Ingestion',
              style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 8),
          const Text('Select SCADA export files from USB media and upload '
              'them securely to the AeroVigil server over HTTPS.'),
          const SizedBox(height: 24),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.usb, color: Color(0xFF2DD4BF)),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(_selectedName ?? 'No file selected',
                            overflow: TextOverflow.ellipsis),
                      ),
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
        ],
      ),
    );
  }
}
