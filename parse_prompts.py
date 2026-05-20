import os

def analyze_prompt(filepath):
    if not os.path.exists(filepath):
        print(f"{filepath} not found")
        return
        
    with open(filepath, "r") as f:
        content = f.read()
        
    print(f"\n--- {os.path.basename(filepath)} ---")
    print(f"Total size: {len(content)} chars (~{len(content)//4} tokens)")
    
    sections = {}
    current_section = "Base Rule / Profile"
    section_start = 0
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('## ') or line.startswith('# ') or line.startswith('### '):
            sections[current_section] = sum(len(l)+1 for l in lines[section_start:i])
            current_section = line.strip()
            section_start = i
            
    sections[current_section] = sum(len(l)+1 for l in lines[section_start:])
    
    for k, v in sections.items():
        if v > 10:
             print(f"  - {k[:40]:<40} : {v} chars")

analyze_prompt("/tmp/spine_prompts/specify_system_prompt.txt")
analyze_prompt("/tmp/spine_prompts/plan_system_prompt.txt")
