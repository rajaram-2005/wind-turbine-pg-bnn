import 'dart:convert';
import 'package:http/http.dart' as http;
class ApiService { ApiService(this.baseUrl); final String baseUrl;
 Future<dynamic> get(String path) async { final r=await http.get(Uri.parse('$baseUrl$path')); if(r.statusCode>=300) throw Exception(r.body); return jsonDecode(r.body); }
 Future<dynamic> post(String path,Object body) async { final r=await http.post(Uri.parse('$baseUrl$path'),headers:{'Content-Type':'application/json'},body:jsonEncode(body)); if(r.statusCode>=300) throw Exception(r.body); return jsonDecode(r.body); }
}
