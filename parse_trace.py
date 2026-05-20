import sys
from collections import defaultdict
from langsmith import Client

def main(trace_id):
    client = Client()
    child_runs = list(client.list_runs(
        project_name=["spine", "spine-or"],
        trace_id=trace_id
    ))
    
    child_runs.sort(key=lambda x: x.start_time if x.start_time else 0)
    llm_runs = [r for r in child_runs if r.run_type == "llm"]
    tool_runs = [r for r in child_runs if r.run_type == "tool"]
    
    read_file_counts = defaultdict(int)
    for t in tool_runs:
         if t.name == "read_file":
              try:
                  inputs = t.inputs or {}
                  if "path" in inputs:
                       read_file_counts[inputs["path"]] += 1
              except Exception:
                  pass
    
    multi_reads = {k: v for k, v in read_file_counts.items() if v >= 3}
    
    valid_research_outputs = 0
    empty_research_outputs = 0
    researcher_file_maps = []
    
    for t in tool_runs:
        if t.name == "task" or "researcher" in t.name.lower():
            out = str(t.outputs)
            if "I'll search broadly" in out or "search broadly" in out or len(out) < 50:
                empty_research_outputs += 1
            else:
                valid_research_outputs += 1
            
            if "file_map" in out and len(out.split("file_map")[1]) > 20:
                researcher_file_maps.append(True)
            else:
                researcher_file_maps.append(False)

    eval_calls = 0
    eval_errors = 0
    eval_success = 0
    redeclared_vars = 0
    used_promise_all = 0
    for t in tool_runs:
        if t.name == "eval":
            eval_calls += 1
            out = str(t.outputs).lower()
            if "error" in out or "traceback" in out:
                eval_errors += 1
                if "redeclaration" in out or "already been declared" in out:
                    redeclared_vars += 1
            else:
                 eval_success += 1
                 
            inp = str(t.inputs).lower()
            if "promise.all" in inp:
                used_promise_all += 1
    
    critic_runs = [r for r in child_runs if "critic" in r.name.lower() and r.run_type == "chain"]
    critic_verdicts = {}
    for r in critic_runs:
        try:
           verdict = r.outputs.get("status", "unknown") if r.outputs else "unknown"
           if verdict != "unknown":
               critic_verdicts[r.name] = verdict
        except Exception:
           pass
           
    print(f"\nFiles read >= 3 times:\n{multi_reads}")
    print(f"Researchers:\n  Valid: {valid_research_outputs}, Empty: {empty_research_outputs}")
    print(f"  File maps populated: {researcher_file_maps}")
    print(f"RLM Eval:\n  Calls: {eval_calls} (Success: {eval_success}, Err: {eval_errors})")
    print(f"  Redeclared vars: {redeclared_vars}")
    print(f"  Promise.all used: {used_promise_all}")
    print(f"Critic verdicts:\n  {critic_verdicts}")
    
    # Save prompts
    import os
    os.makedirs("/tmp/spine_prompts", exist_ok=True)
    saved_specify = False
    saved_plan = False
    
    for r in llm_runs:
        phase = "unknown"
        curr = r
        parent_map = {p.id: p for p in child_runs}
        while curr.parent_run_id:
            curr = parent_map.get(curr.parent_run_id)
            if not curr: break
            name = curr.name.lower()
            if name in ["specify", "plan", "critic_specify", "critic_plan"]:
                phase = name
                break
        if phase == "unknown":
            for tag in (r.tags or []):
                 if tag in ["specify", "plan"]:
                      phase = tag
                      break
                      
        if phase in ["specify", "plan"]:
             if phase == "specify" and saved_specify: continue
             if phase == "plan" and saved_plan: continue
             
             if r.inputs and "messages" in r.inputs:
                  msgs = r.inputs["messages"]
                  for msg in msgs:
                       content = ""
                       if isinstance(msg, dict):
                            if msg.get("id", []) and msg["id"][-1] == "SystemMessage":
                                 content = msg.get("kwargs", {}).get("content", "")
                            elif msg.get("type") == "system":
                                 content = msg.get("content", "")
                       if content:
                            with open(f"/tmp/spine_prompts/{phase}_system_prompt.txt", "w") as f:
                                 f.write(content)
                            if phase == "specify":
                                saved_specify = True
                            if phase == "plan":
                                saved_plan = True

if __name__ == "__main__":
    if len(sys.argv) > 1:
         main(sys.argv[1])
    else:
         main("019e42d1-5dae-79f3-ae0e-9761df7c86e7")
