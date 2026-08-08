"""AeroVigilAI gateway telemetry agent (pluggable protocol skeleton)."""
from __future__ import annotations
import abc, json, sqlite3, time
from datetime import datetime, timezone
from typing import Any
from urllib.request import Request, urlopen

class ProtocolConnector(abc.ABC):
 @abc.abstractmethod
 def poll(self) -> dict[str, Any]: ...
class HTTPSPollingConnector(ProtocolConnector):
 def __init__(self,url:str): self.url=url
 def poll(self):
  with urlopen(self.url,timeout=10) as r: return json.loads(r.read())
class MQTTConnector(ProtocolConnector):
 def __init__(self,*args,**kwargs): self.args=args; self.kwargs=kwargs
 def poll(self): raise NotImplementedError("Install paho-mqtt and implement broker subscription for this gateway")
class ModbusTCPConnector(ProtocolConnector):
 def __init__(self,*args,**kwargs): self.args=args; self.kwargs=kwargs
 def poll(self): raise NotImplementedError("Install pymodbus and configure holding-register mapping")
class OPCUAConnector(ProtocolConnector):
 def __init__(self,*args,**kwargs): self.args=args; self.kwargs=kwargs
 def poll(self): raise NotImplementedError("Install asyncua and configure node IDs")
def normalize(raw:dict[str,Any], asset_id:str)->dict[str,Any]:
 keys=("vibration_mms","temperature_c","rpm","oil_viscosity_cst","load_pct")
 return {"asset_id":asset_id,"timestamp":datetime.now(timezone.utc).isoformat(),"telemetry":{k:float(raw[k]) for k in keys}}
class GatewayAgent:
 def __init__(self,connector:ProtocolConnector,server_url:str,asset_id:str,db_path="hardware_buffer.sqlite"):
  self.connector,self.url,self.asset_id=connector,server_url.rstrip("/"),asset_id; self.db=sqlite3.connect(db_path); self.db.execute("create table if not exists buffer (payload text)")
 def send(self,payload):
  data=json.dumps(payload).encode(); req=Request(self.url+"/api/hardware/stream",data=data,headers={"Content-Type":"application/json"},method="POST")
  try:
   with urlopen(req,timeout=10): pass
   return True
  except OSError:
   self.db.execute("insert into buffer values(?)",(data.decode(),)); self.db.commit(); return False
 def flush(self):
  for rowid,payload in self.db.execute("select rowid,payload from buffer").fetchall():
   if self.send(json.loads(payload)): self.db.execute("delete from buffer where rowid=?",(rowid,)); self.db.commit()
 def run_forever(self,interval=10):
  while True:
   self.flush(); self.send(normalize(self.connector.poll(),self.asset_id)); time.sleep(interval)
