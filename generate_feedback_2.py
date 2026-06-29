#!/usr/bin/env python3
from openai import OpenAI
from dotenv import load_dotenv
import re
from pathlib import Path
import os, json, sqlite3
from typing import List, Dict, Optional, Tuple
import tempfile
from collections import Counter

SCRIPT_DIR    = Path(__file__).parent.resolve()          
LOCAL_AG      = SCRIPT_DIR / "local_autograder"
WEBSITE_DIR   = SCRIPT_DIR / "website"
SUBMISSIONS_ROOT = WEBSITE_DIR / "Submissions"          

db_path = LOCAL_AG / "tpch.sqlite"
assignment = SCRIPT_DIR / "assignment.txt"
feedback_json = SCRIPT_DIR / "feedback.json"
error_bank_path = SCRIPT_DIR / "error_bank.json"
student_all_errors_path = SCRIPT_DIR / "student_feedback_total.json"

MODEL = "gpt-4o-mini"   
MAX_ASSIGNMENT_CHARS = 1200
MAX_CASES_PER_SUB = 50
MAX_CHARS_PER_OUTPUT = 400
MAX_CODE_CHARS = 6000

def read_excerpt(p: Path, limit: int) -> str:
    if not p.exists():
        return ""
    return p.read_text(errors="ignore")[:limit].strip()

def condense(msg: str, limit: int) -> str:
    if not msg:
        return ""
    one = " ".join(line.strip() for line in msg.splitlines() if line.strip())
    return one.replace("\\\\", "\\")[:limit]

def fetch_grouped_outputs(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    q = """
    SELECT submission_id, COALESCE(output,'')
    FROM results_filtered
    ORDER BY submission_id;
    """
    grouped: Dict[str, List[str]] = {}
    for sid, output in conn.execute(q):
        grouped.setdefault(sid, []).append(output or "")
    return grouped

def get_student_name_from_submission_id(submission_id: str) -> str:
    """
    Now that submission folders are named after students (e.g. Harshini_Nujella),
    convert folder name back to full name by replacing underscores with spaces.
    """
    return submission_id.replace("_", " ")

def find_submission_dir(submission_id: str) -> Optional[Path]:
    """
    Look for the student's submission folder in website/Submissions/
    Folder is named after the student e.g. Harshini_Nujella
    """
    direct = SUBMISSIONS_ROOT / submission_id
    if direct.exists() and direct.is_dir():
        return direct

    # Try with spaces replaced by underscores
    underscore = SUBMISSIONS_ROOT / submission_id.replace(" ", "_")
    if underscore.exists() and underscore.is_dir():
        return underscore

    # Fallback — scan all folders
    for p in SUBMISSIONS_ROOT.iterdir():
        if p.is_dir() and submission_id.lower() in p.name.lower():
            return p

    return None

def read_student_code_excerpt(sub_dir: Optional[Path], limit: int = MAX_CODE_CHARS) -> str:
    if not sub_dir or not sub_dir.exists():
        return ""
    pieces, total = [], 0
    py_files = sorted(sub_dir.glob("*.py"))
    for py in py_files:
        header = f"# FILE: {py.name}\n"
        try:
            code = py.read_text(errors="ignore")
        except Exception:
            code = ""
        chunk = header + code
        if total + len(chunk) > limit:
            chunk = chunk[: max(0, limit - total)]
        pieces.append(chunk)
        total += len(chunk)
        if total >= limit:
            break
    return ("\n".join(pieces)).strip()

def load_error_bank(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def classify_errors_with_bank(raw_messages: List[str], bank: Dict[str, dict]) -> List[str]:
    codes = []
    for i in raw_messages:
        matched = False
        for code, entry in bank.items():
            if code == "UNCLASSIFIED":
                continue
            for pat in entry.get("regex", []):
                try:
                    if re.search(pat, i, re.I | re.S):
                        codes.append(code)
                        matched = True
                        break
                except re.error:
                    continue
            if matched:
                break
        if not matched:
            codes.append("UNCLASSIFIED")
    seen, out = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out or ["UNCLASSIFIED"]

def make_student_pattern_block_for_submission(bank: Dict[str, dict], error_codes: List[str]) -> str:
    if not error_codes:
        return "none"
    by_category: Dict[str, List[str]] = {}
    for i in error_codes:
        category = bank.get(i, {}).get("category", "Unclassified")
        by_category.setdefault(category, []).append(i)
    lines = []
    for category, codes in by_category.items():
        lines.append(f"{category}: {', '.join(sorted(codes))}")
    return "\n".join(lines)

HUMANIZE = {
    "TYPE_NONE": "mixing None with numbers/operations",
    "FLOAT_ON_NONE": "calling float() on None",
    "OUTPUT_MISMATCH": "output formatting mismatch",
    "MATH_OPS_INCORRECT": "incorrect math operations",
    "LOOP_NEVER_UPDATES": "loop condition never updates (possible infinite loop)",
    "LOOP_BAD_RANGE": "range() bounds/off-by-one",
    "FUNC_ARGS_ORDER": "function argument order/count wrong",
    "FUNC_MISSING_RETURN":"function missing return value",
    "TYPE_MIX_STR_INT": "mixing string and int types",
    "FILE_PATH_ERROR": "file path not found",
    "MISSING_RETURN": "function prints instead of returning a value",
    "WRONG_FORMULA": "wrong formula or calculation logic",
    "NONE_ARITHMETIC": "using None in arithmetic operations",
    "WRONG_RESISTOR_DECODE":"resistor color code decoded incorrectly",
    "INFINITE_LOOP": "infinite loop or too many input() calls",
    "MISSING_OUTPUT_HEADING": "missing required task heading or output line",
    "PROMPT_WORDING_MISMATCH":"output text doesn't match expected wording",
    "STRING_FLOAT_CRASH": "converting non-numeric string to float/int",
    "LEN_NONE_ERROR": "calling len() or indexing on None",
    "WRONG_RETURN_TYPE": "function returns wrong type (e.g. None instead of float)",
    "MISSING_ERROR_HANDLING":"missing ValueError or exception handling",
    "MISSING_BAND_INSTRUCTION":"missing required user instruction text",
    "MISSING_OPTIONAL_PARAM":"function parameter not made optional",
    "HARDCODED_FILE_PATH": "hardcoded local file path (won't work on autograder)",
    "FILE_READING_ERROR": "reading file header row as data",
    "UNDEFINED_VARIABLE": "variable used before it was defined",
    "LOOP_COUNT_ERROR": "loop counter or limit condition logic wrong",
    "WRONG_MULTIMETER_RANGE": "wrong multimeter range boundary logic",
    "WRONG_MULTIPLIER": "color code multiplier uses 10*n instead of 10**n",
    "MISSING_DEPENDENCY": "missing import or module not found",
    "SYNTAX_ERROR": "syntax or indentation error in code"
}

def load_student_errors(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
        return {}
    except json.JSONDecodeError:
        return {}

def save_student_errors(path: Path, data: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(json.dumps(data, indent=2))
        temp_name = tmp.name
    Path(temp_name).replace(path)

def record_student_submission(
    totals: Dict[str, dict],
    full_name: str,
    submission_id: str,
    error_codes: List[str],
    feedback_text: Optional[str] = None
) -> Dict[str, dict]:
    student = totals.setdefault(full_name, {
        "full_name":       full_name,
        "feedback_count":  0,
        "history":         []
    })
    student["feedback_count"] += 1
    student["history"].append({
        "submission_id": submission_id,
        "error_codes": error_codes,
        "feedback": feedback_text or ""
    })
    totals[full_name] = student
    return totals

def detect_improvements_and_regressions(
    student: dict,
    now_codes: List[str],
    window: int = 3
) -> Tuple[List[str], List[str], List[str]]:
    """
    Compare current errors against recent history.
    Returns: fixed, recurring, new_codes
    """
    hist = student.get("history", [])
    if not hist:
        return [], [], now_codes

    # Get error codes from last submissions
    recent_submissions = hist[-window:]
    past_codes_per_sub = [set(h.get("error_codes", [])) for h in recent_submissions]

    # recurring codes
    if past_codes_per_sub:
        recurring = set.intersection(*past_codes_per_sub) if len(past_codes_per_sub) > 1 else past_codes_per_sub[0]
    else:
        recurring = set()

    # fixed/improved codes
    last_codes = set(hist[-1].get("error_codes", [])) if hist else set()
    fixed = list(last_codes - set(now_codes))

    # new codes
    all_past = set().union(*past_codes_per_sub) if past_codes_per_sub else set()
    new_codes = list(set(now_codes) - all_past)

    # recurring codes that are still present now
    recurring_now = list(set(now_codes) & recurring)

    return fixed, recurring_now, new_codes

def build_rag_examples(student: dict, max_examples: int = 3) -> str:
    """
    Pull the most recent feedback messages from the student's history
    to use as RAG examples — shows the AI what good feedback looks like
    for THIS student.
    """
    hist = student.get("history", [])
    if not hist:
        return "(no prior feedback for this student)"

    examples = []
    for h in reversed(hist[-max_examples:]):
        fb = h.get("feedback", "").strip()
        codes = h.get("error_codes", [])
        if fb:
            humanized = [HUMANIZE.get(c, c) for c in codes if c != "UNCLASSIFIED"]
            label = ", ".join(humanized) if humanized else "general error"
            examples.append(f"- Error type: {label}\n  Feedback given: \"{fb}\"")

    if not examples:
        return "(no prior feedback for this student)"

    return "\n".join(examples)

def summarize_history(
    student: dict,
    now_codes: List[str],
    top_k: int = 3,
    window: int = 5
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Returns history summary string + fixed/recurring/new code lists.
    """
    hist = student.get("history", [])
    fixed, recurring, new_codes = detect_improvements_and_regressions(
        student, now_codes, window=3
    )

    if not hist:
        return "First submission — no prior history.", fixed, recurring, new_codes

    life = Counter()
    for h in hist:
        life.update(set(h.get("error_codes", [])))

    top  = [f"{c}({life[c]})" for c, _ in life.most_common(top_k)]
    lines = [
        f"Total prior submissions: {len(hist)}",
        ("Most frequent past errors: " + ", ".join(top)) if top else "Most frequent past errors: (none)",
        ("Current errors: " + ", ".join([HUMANIZE.get(c, c) for c in now_codes])) if now_codes else "Current errors: (none)",
    ]

    if fixed:
        humanized_fixed = [HUMANIZE.get(c, c) for c in fixed]
        lines.append("IMPROVED since last submission: " + ", ".join(humanized_fixed))
    if recurring:
        humanized_recurring = [HUMANIZE.get(c, c) for c in recurring]
        lines.append("RECURRING errors (seen multiple times): " + ", ".join(humanized_recurring))
    if new_codes:
        humanized_new = [HUMANIZE.get(c, c) for c in new_codes]
        lines.append("NEW errors this submission: " + ", ".join(humanized_new))

    return "\n".join(lines), fixed, recurring, new_codes



def detect_common_code_issues(student_code):
    issues = [];

    if not student_code or student_code == "(none)":
        return issues

    compact_code = student_code.replace(" ", "")

    #code = student_code.lower()

    if "^" in student_code:
        issues.append(
            "Possible operator issue: code uses `^`, which is bitwise XOR in Python, not exponentiation or comparison."
        )

    if "10*color_values" in compact_code:
        issues.append(
            "Possible color-band multiplier issue: code appears to multiply by 10*n instead of using 10**n."
        )

    if "class Parallel" in student_code:
        parallel_section = student_code.split("class Parallel", 1)[1]
        if "total_resistance+=i.get_resistance()" in compact_code and "1/" not in parallel_section.replace(" ", ""):
            issues.append(
                "Possible parallel resistance issue: Parallel.get_resistance() may be adding resistances directly instead of using reciprocal sum."
            )

    if "def get_resistance" in student_code and "return self.resistance" not in student_code:
        issues.append(
            "Possible get_resistance issue: method may not be returning the stored resistance value."
        )

    return issues

System_prompt = (
"""You are a warm, encouraging teaching assistant for an introductory Python programming course in Electrical Engineering.

Your job is to give ONE short, human-like feedback message per student based on their autograder results.
- failed autograder tests
- student code
- assignment instructions
- detected code patterns
- student history, if available

### Core Goal ###
Give accurate feedback that is specific and grounded in the evidence provided. Do not guess/invent errors.

### Tone Guidelines ###
- Sound like a real TA, not a bot — warm, direct, and encouraging
- Balance emotional support with practical, specific guidance
- Sometimes acknowledge improvement when it exists — students work hard and deserve recognition
- Be specific about what to fix, but never reveal the full solution
- Keep it concise: 15-40 words total
- Use natural language, not bullet points
- NEVER mention internal labels like UNCLASSIFIED, WRONG_FORMULA, TYPE_NONE etc.

### Accuracy Rules ###
- Use failed tests and detected code patterns as the basis for your feedback
- If multiple major issue areas exist, prioritize the top 1-2 most critical ones based on the failed tests.
- Do not focus only on the first failed test if other detected issues are more critical.
- If a detected pattern conflicts with the failed tests, prioritize the failed tests but also consider the context of the detected pattern.
- If the student fixed something from a previous submission, acknowledge that improvement.
- Do not say a dictionary is missing values unless the actual issue is missing color keys.
- If the multiplier uses 10*n instead of 10**n, call it a color-multiplier issue, not just a dictionary issue.
- If failed tests mention specific inputs like brown, red, or orange, do NOT assume those color dictionary values are wrong. First check the student code. If the dictionary maps colors correctly but the multiplier uses 10*n instead of 10**n, describe it as a third color-band multiplier formula issue.
- When static code patterns identify multiple issues, compare them against the assignment goals and choose the most important 1-2 issues. For this assignment, parallel resistance and color-band resistance calculation are both major concepts.

### Examples of good feedback ###
- "Great progress — your parallel formula is now correct! Take another look at set_resistance() though, the color-band multiplier should use 10**n not 10*n."
- "You're close! The multimeter range logic in get_multimeter_range() isn't handling all ranges — double-check your loop covers every case in the available list."
- "Nice work on the series resistance! compute_power() seems to be printing the result instead of returning it."
- "We've noticed set_resistance() keeps getting the multiplier wrong — it's worth reviewing how the third color band maps to a power of 10."

### Improvement Acknowledgment Rules ###
- If the student FIXED errors from last submission → start with genuine praise e.g.:
  "Great progress! Your [specific thing] improved — now focus on fixing [current issue]."
- If the student has RECURRING errors → gently but clearly flag it e.g.:
  "We've noticed [error type] keeps coming up — it might be worth revisiting that concept."
- If this is their FIRST submission → be encouraging and specific about what to fix
- If they have NEW errors → mention them clearly but kindly

### RAG Examples ###
Below are examples of good feedback previously given to this student.
Use these as style/tone reference — do NOT repeat them verbatim:
{rag_examples}

### Output Format ###
Return ONLY a valid JSON object, no text before or after:
{{
  "submission_id": "<submission_id>",
  "feedback": [{{"message": "<your feedback here>"}}]
}}
"""
)

template = """
Course: Intro Python

Assignment brief:
{assignment_excerpt}

Failed tests:
{outputs_block}

Detected code patterns:
{student_pattern_block}

Student history:
{student_history_block}

Student submission excerpt:
{code_excerpt}

Task:
Write ONE feedback message for this student.

Before writing, silently decide:
1. What are the main failed concept areas?
2. Are there multiple major issues?
3. Did the student improve anything from a previous submission?
4. What is the most useful next step?

Feedback requirements:
- 20-50 words
- Mention the most important 1-2 issues only
- Be warm and specific
- Do not reveal the full solution
- Do not mention internal labels
- Do not invent issues not supported by the failed tests/code patterns

Return JSON exactly in this format:
{{
  "submission_id": "{submission_id}",
  "feedback": [{{"message": "<20-50 word encouraging feedback>"}}]
}}
"""

def generate_feedback():
    load_dotenv()
    client = OpenAI()

    assignment_excerpt = read_excerpt(assignment, MAX_ASSIGNMENT_CHARS)
    error_bank = load_error_bank(error_bank_path)
    student_totals = load_student_errors(student_all_errors_path)

    conn = sqlite3.connect(db_path)
    outputs_by_sid = fetch_grouped_outputs(conn)
    conn.close()

    results_out = []

    for submission_id, outputs in outputs_by_sid.items():
        full_name = get_student_name_from_submission_id(submission_id)

        condensed = [condense(o, MAX_CHARS_PER_OUTPUT) for o in outputs[:MAX_CASES_PER_SUB]]
        raw_errors = condensed[:]
        error_codes = classify_errors_with_bank(raw_errors, error_bank)

        student_profile = student_totals.get(full_name, {})
        history_block, fixed, recurring, new_codes = summarize_history(
            student=student_profile,
            now_codes=error_codes,
            top_k=3,
            window=5
        )
        rag_examples  = build_rag_examples(student_profile, max_examples=3)
        #pattern_block = make_student_pattern_block_for_submission(error_bank, error_codes)

        #Find student's code in new Submissions folder
        sub_dir = find_submission_dir(submission_id)
        code_excerpt = read_student_code_excerpt(sub_dir, MAX_CODE_CHARS) or "(none)"

        bank_pattern_block = make_student_pattern_block_for_submission(error_bank, error_codes)
        
        static_issues = detect_common_code_issues(code_excerpt)
        static_pattern_block = (
            "\n".join(f"- {issue}" for issue in static_issues)
            if static_issues
            else "- No high-confidence static code patterns detected."
        )

        pattern_block = f"""
        Autograder/error-bank categories:
        {bank_pattern_block}

        Static code patterns:
        {static_pattern_block}
        """.strip()
        
        outputs_block = "\n- ".join([""] + condensed) if condensed else "(none)"

        user_prompt = template.format(
            outputs_block=outputs_block,
            assignment_excerpt=assignment_excerpt or "(none)",
            submission_id=submission_id,
            code_excerpt=code_excerpt,
            student_pattern_block=pattern_block,
            student_history_block=history_block
        )

        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": System_prompt.format(rag_examples=rag_examples)},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            text = resp.output_text
            parsed = json.loads(text)
            fb_list = parsed.get("feedback", [])
            msg = (fb_list[0].get("message", "").strip() if fb_list else "")

            student_totals = record_student_submission(
                student_totals,
                full_name=full_name,
                submission_id=submission_id,
                error_codes=error_codes,
                feedback_text=msg if msg else None
            )

            if msg:
                results_out.append({
                    "submission_id": submission_id,
                    "full_name": full_name,
                    "feedback": msg
                })
                print(f"{submission_id}: {msg}")
            else:
                print(f"No feedback for {submission_id}")

        except Exception as e:
            print(f"Failed for {submission_id}: {e}\n{text[:400] if 'text' in dir() else ''}")

    save_student_errors(student_all_errors_path, student_totals)

    existing = []
    if feedback_json.exists():
        try:
            existing = json.loads(feedback_json.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    merged = {}
    for row in existing:
        merged[row["submission_id"]] = row
    for row in results_out:
        merged[row["submission_id"]] = row

    feedback_json.write_text(json.dumps(list(merged.values()), indent=2))
    print(f"\nWrote to feedback.json with {len(merged)} records.")

if __name__ == "__main__":
    generate_feedback()