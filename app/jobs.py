"""Minimal in-process asynchronous job manager for trusted framework actions."""
from __future__ import annotations
import asyncio, uuid
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED = {"train", "evaluate", "export", "active-sample", "explain"}
@dataclass
class Job:
    status: str = "Pending"; logs: list[str] = field(default_factory=list); task: asyncio.Task | None = None
class JobManager:
 def __init__(self): self.jobs: dict[str, Job] = {}
 def submit(self, job_type: str, args: list[str]) -> str:
  if job_type not in ALLOWED: raise ValueError(f"unsupported job type: {job_type}")
  if any(not isinstance(x,str) or x.startswith("-") and x not in {"--config","--checkpoint","--out","--epochs"} for x in args): raise ValueError("unsupported job argument")
  jid=uuid.uuid4().hex; job=Job(); self.jobs[jid]=job; job.task=asyncio.create_task(self._run(job,job_type,args)); return jid
 async def _run(self, job: Job, job_type: str, args: list[str]):
  job.status="Running"; job.logs.append(f"queued physics {job_type}")
  process=await asyncio.create_subprocess_exec("python","-m","src","physics",job_type,*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
  assert process.stdout
  async for line in process.stdout:
   job.logs.append(line.decode(errors="replace").rstrip())
   del job.logs[:-100]
  code=await process.wait(); job.status="Completed" if code==0 else "Failed"; job.logs.append(f"exit code {code}")
