"""
Convert paradigm_scenario markdown reports to JSON format.
Handles multiple MD formatting styles:
  - ## Header sections
  - **Bold Key:** Value inline pairs
  - | Table | Format |
  - Numbered bold items (1. **Key:** Value)
  - Mixed formats
"""
import json
import re
import os
import glob


SECTION_ALIASES = {
    'use case name': 'use_case_name',
    'use case': 'use_case_name',
    'description': 'description',
    'primary actor': 'primary_actor',
    'primary actors': 'primary_actor',
    'problem domain context': 'context',
    'context': 'context',
    'preconditions': 'preconditions',
    'precondition': 'preconditions',
    'postconditions': 'postconditions',
    'postcondition': 'postconditions',
    'main flow': 'main_flow',
    'basic flow': 'main_flow',
    'normal flow': 'main_flow',
    'alternative flows': 'alternative_flows',
    'alternative flow': 'alternative_flows',
    'exceptions': 'exception_flows',
    'exception flows': 'exception_flows',
    'exception': 'exception_flows',
}


def normalize_key(raw):
    """Map a raw section heading to a canonical key."""
    cleaned = raw.strip().rstrip(':').strip()
    cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
    cleaned = cleaned.replace('**', '').strip()
    cleaned_lower = cleaned.lower()
    for alias, canonical in SECTION_ALIASES.items():
        if alias in cleaned_lower:
            return canonical
    return None


def parse_md(md_text):
    """
    Parse markdown text into sections. Handles:
      1. ## or # headers
      2. **Bold Key:** value or **Bold Key:**: value (colon inside/outside bold)
      3. | **Key** | value | table rows
      4. Numbered bold items: 1. **Key:** value
      5. **Bold Key** (value on next line)
    """
    sections = {}
    current_key = None
    current_lines = []

    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped == '---':
            if current_key is not None:
                current_lines.append(line)
            i += 1
            continue

        new_key = None
        new_val = None
        handled = False

        # 1. Check for markdown headers: # or ##
        header_match = re.match(r'^#{1,4}\s+(?:\d+\.?\s*)?(.*)', line)
        if header_match:
            raw_header = header_match.group(1).strip()
            key = normalize_key(raw_header)
            if key:
                new_key = key
                # Extract inline value for "## Use Case: Name" pattern
                if key == 'use_case_name' and ':' in raw_header:
                    val = raw_header.split(':', 1)[1].strip().replace('**', '')
                    if val:
                        new_val = val
            handled = True

        # 2. Check for table rows: | **Key** | Value |
        if not handled:
            table_match = re.match(r'^\|\s*\*\*(.+?)\*\*\s*\|\s*(.*?)\s*\|', stripped)
            if table_match:
                tkey = normalize_key(table_match.group(1).strip())
                if tkey:
                    new_key = tkey
                    tval = table_match.group(2).strip()
                    if tval:
                        new_val = tval
                handled = True

        # Skip table header separator
        if not handled and re.match(r'^\|[-\s|]+\|$', stripped):
            handled = True

        # 3. Check for bold key-value pairs: **Key:** Value (colon inside bold)
        if not handled:
            bold_kv_inside = re.match(r'^\*\*(.+?):\*\*\s*(.*)', stripped)
            if bold_kv_inside:
                rk = bold_kv_inside.group(1).strip()
                rv = bold_kv_inside.group(2).strip()
                key = normalize_key(rk)
                if key:
                    new_key = key
                    if rv:
                        new_val = rv
                    handled = True

        # 3b. Check for bold key-value pairs: **Key**: Value (colon outside bold)
        if not handled:
            bold_kv_outside = re.match(r'^\*\*(.+?)\*\*\s*:\s*(.*)', stripped)
            if bold_kv_outside:
                rk = bold_kv_outside.group(1).strip()
                rv = bold_kv_outside.group(2).strip()
                key = normalize_key(rk)
                if key:
                    new_key = key
                    if rv:
                        new_val = rv
                    handled = True

        # 3c. Bold key without colon: **Key** (value on next line)
        if not handled:
            bold_key_only = re.match(r'^\*\*(.+?)\*\*\s*$', stripped)
            if bold_key_only:
                rk = bold_key_only.group(1).strip()
                key = normalize_key(rk)
                if key:
                    new_key = key
                    handled = True

        # 4. Numbered bold keys: 1. **Key:** Value or 1. **Key**: Value
        if not handled:
            num_bold = re.match(r'^\d+\.\s*\*\*(.+?)[:]*\*\*\s*[:]*\s*(.*)', stripped)
            if num_bold:
                rk = num_bold.group(1).strip().rstrip(':')
                rv = num_bold.group(2).strip()
                key = normalize_key(rk)
                if key:
                    new_key = key
                    if rv:
                        new_val = rv
                    handled = True

        # Apply new section or append to current
        if new_key:
            # Flush previous section
            if current_key is not None:
                sections[current_key] = '\n'.join(current_lines).strip()
            current_key = new_key
            current_lines = []
            if new_val:
                current_lines.append(new_val)
        elif not handled and current_key is not None:
            current_lines.append(line)

        i += 1

    # Final flush
    if current_key is not None:
        sections[current_key] = '\n'.join(current_lines).strip()

    return sections


def clean_text(text):
    """Remove excessive whitespace and markdown artifacts."""
    text = text.strip()
    text = text.replace('<br>', ' ')
    return text


def extract_use_case_name(sections):
    val = sections.get('use_case_name', '').strip()
    val = val.replace('**', '').strip()
    val = val.strip('"').strip("'")
    return val if val else "Unknown"


def extract_description(sections):
    val = sections.get('description', '').strip()
    val = val.replace('**', '').strip()
    return clean_text(val)


def extract_primary_actor(sections):
    val = sections.get('primary_actor', '').strip()
    val = val.replace('**', '').strip()
    if not val:
        return ["Unknown"]
    actors = [a.strip() for a in re.split(r'[,\n]', val) if a.strip()]
    return actors if actors else ["Unknown"]


def extract_context(sections):
    val = sections.get('context', '').strip()
    if not val:
        return ""
    lines = val.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and line != '---':
            cleaned.append(line)
    result = ' '.join(cleaned)
    result = result.replace('<br>', ' ')
    return result


def parse_list_items(text):
    """Parse a section into list items (numbered, bulleted, or paragraph)."""
    text = text.strip()
    if not text or text.lower() in ('none.', 'none'):
        return []

    items = []
    current = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^(\d+\.\s+|- |\* )', stripped):
            if current:
                items.append(' '.join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        items.append(' '.join(current))

    return items if items else [text]


def extract_preconditions(sections):
    val = sections.get('preconditions', '').strip()
    if not val or val.lower() in ('none.', 'none'):
        return ["None"]
    items = parse_list_items(val)
    return items if items else ["None"]


def extract_postconditions(sections):
    val = sections.get('postconditions', '').strip()
    if not val or val.lower() in ('none.', 'none'):
        return ["None"]
    items = parse_list_items(val)
    return items if items else ["None"]


def extract_flow_steps(sections, key):
    """Extract numbered flow steps from a section."""
    val = sections.get(key, '').strip()
    if not val or val.lower() in ('none.', 'none'):
        return []

    steps = []
    current_step = []

    for line in val.splitlines():
        stripped = line.strip()
        if not stripped or stripped == '---':
            continue
        if re.match(r'^\d+\.\s+', stripped):
            if current_step:
                steps.append(' '.join(current_step))
            current_step = [stripped]
        else:
            current_step.append(stripped)

    if current_step:
        steps.append(' '.join(current_step))

    return steps


def extract_block_items(sections, key):
    """Extract alternative or exception flow blocks."""
    val = sections.get(key, '').strip()
    if not val or val.lower() in ('none.', 'none'):
        return []

    items = []
    current_block = []

    for line in val.splitlines():
        stripped = line.strip()
        if not stripped or stripped == '---':
            continue

        is_new_block = False
        # Numbered bold: "1. **..."
        if re.match(r'^\d+\.\s+\*\*', stripped):
            is_new_block = True
        # Dash bold: "- **..."
        elif re.match(r'^-\s+\*\*', stripped):
            is_new_block = True
        # Standalone bold: "**A1..." or "**E1..."
        elif re.match(r'^\*\*[AEae]\w*[\s\d(]', stripped):
            is_new_block = True
        # "(Step X)" pattern
        elif re.match(r'^\*\*\(Step', stripped):
            is_new_block = True
        # Bold heading with dash: "**Something** –"
        elif re.match(r'^\*\*[^*]+\*\*\s*[-–]', stripped):
            is_new_block = True
        # Numbered with period and text  (not sub-numbers like 1.1)
        elif re.match(r'^\d+\.\s+\*', stripped):
            is_new_block = True

        if is_new_block:
            if current_block:
                items.append(' '.join(current_block))
            current_block = [stripped]
        else:
            current_block.append(stripped)

    if current_block:
        items.append(' '.join(current_block))

    return items


def infer_area(context):
    """Infer the domain area from the context."""
    ctx = context.lower()
    if any(w in ctx for w in ['traffic', 'mobility', 'congestion', 'urban']):
        return "Urban Mobility"
    if any(w in ctx for w in ['sensor', 'heat map', 'environmental', 'simulator', 'wonderland']):
        return "Environmental Monitoring"
    if any(w in ctx for w in ['patient', 'medical', 'doctor', 'prescription', 'healthcare',
                               'clinical', 'health', 'wearable', 'nurse', 'pharmacist',
                               'lab specialist', 'digi', 'uhope', 'electronic medical record',
                               'gdpr', 'appointment', 'billing']):
        return "Healthcare"
    if any(w in ctx for w in ['league', 'team', 'match', 'referee', 'sport', 'football',
                               'ifa', 'fan', 'budget report']):
        return "Sports Management"
    if any(w in ctx for w in ['dataset', 'research', 'archive', 'metadata', 'repository',
                               'depositor']):
        return "Research Data Management"
    return "General"


def md_to_json(md_text, uc_index, filename_hint=""):
    """Convert a markdown use case report to JSON."""
    sections = parse_md(md_text)

    use_case_name = extract_use_case_name(sections)
    # Fallback: derive name from filename
    if use_case_name == "Unknown" and filename_hint:
        name_from_file = filename_hint.replace('_report.md', '').replace('_', ' ').strip()
        if name_from_file:
            use_case_name = name_from_file
    description = extract_description(sections)
    # Fallback: if description is empty, try to derive from use case name
    if not description and use_case_name != "Unknown":
        description = use_case_name
    primary_actors = extract_primary_actor(sections)
    context = extract_context(sections)
    preconditions = extract_preconditions(sections)
    postconditions = extract_postconditions(sections)
    main_flow = extract_flow_steps(sections, 'main_flow')
    alternative_flows = extract_block_items(sections, 'alternative_flows')
    exception_flows = extract_block_items(sections, 'exception_flows')
    area = infer_area(context)

    uc_id = f"UC-{uc_index:03d}"
    triggering_event = f"{primary_actors[0]} initiates '{use_case_name}'."

    return {
        "use_case_name": use_case_name,
        "unique_id": uc_id,
        "area": area,
        "context_of_use": context,
        "scope": "Target System",
        "level": "User-goal",
        "primary_actors": primary_actors,
        "supporting_actors": ["System"],
        "stakeholders_and_interests": [
            f"{primary_actors[0]}: Achieve the goal described in the use case."
        ],
        "description": description,
        "triggering_event": triggering_event,
        "trigger_type": "External",
        "preconditions": preconditions,
        "postconditions": postconditions,
        "assumptions": [
            "The actor has appropriate access to the system."
        ],
        "requirements_met": [
            description if description else use_case_name
        ],
        "priority": "Medium",
        "risk": "Medium",
        "outstanding_issues": [
            "None identified."
        ],
        "main_flow": main_flow,
        "alternative_flows": alternative_flows,
        "exception_flows": exception_flows,
        "information_for_steps": main_flow.copy()
    }


def process_folder(folder_path):
    """Process all MD files in a folder, overwriting existing JSON (except folder 1)."""
    md_files = sorted(glob.glob(os.path.join(folder_path, '*_report.md')))
    if not md_files:
        print(f"  No _report.md files found")
        return

    folder_name = os.path.basename(folder_path)

    for idx, md_file in enumerate(md_files, start=1):
        json_file = md_file.replace('_report.md', '_report.json')
        basename = os.path.basename(json_file)

        if folder_name == '1':
            print(f"  SKIP (folder 1): {basename}")
            continue

        with open(md_file, 'r', encoding='utf-8') as f:
            md_text = f.read()

        uc_json = md_to_json(md_text, idx, filename_hint=os.path.basename(md_file))

        issues = []
        if uc_json['use_case_name'] == 'Unknown':
            issues.append('name=Unknown')
        if not uc_json['main_flow']:
            issues.append('empty main_flow')
        if not uc_json['description']:
            issues.append('empty desc')
        if uc_json['area'] == 'General':
            issues.append('area=General')

        status = "OK" if not issues else f"WARN({', '.join(issues)})"

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(uc_json, f, indent=2, ensure_ascii=False)

        print(f"  {status}: {basename}")


def main():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paradigm_scenario')

    for folder_name in sorted(os.listdir(base_dir), key=lambda x: int(x) if x.isdigit() else 999):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"\n=== Folder: {folder_name} ===")
            process_folder(folder_path)


if __name__ == '__main__':
    main()
