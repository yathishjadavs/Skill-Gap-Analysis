"""Stage 7 + Recommendation Stage 2: Gemini LLM abstraction (DDD 6, 7, 9).
 
Exactly two LLM calls per analysis are made through this module:
  1. extract_skills()   - skill extraction, categorization & leveling (temp 0.1)
  2. rerank_courses()   - personalized course re-ranking (temp 0.2)
 
Both use the exact prompt templates from DDD Section 7. Responses are parsed as
strict JSON with a single stricter-prompt retry and a regex-extraction fallback.
Rate limits (429) are retried with exponential backoff (1s, 2s, 4s; max 3).
"""
from __future__ import annotations
 
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
 
from app.config import get_settings
from app.constants.categories import VALID_CATEGORIES
 
logger = logging.getLogger(__name__)
 
MAX_BACKOFF_RETRIES = 3
BACKOFF_SCHEDULE = [1, 2, 4]  # seconds
FALLBACK_CATEGORY = "soft_skills"
 
 
# ---------------------------------------------------------------------------
# Typed errors (mapped to HTTP codes at the route layer)
# ---------------------------------------------------------------------------
class LLMError(Exception):
    pass
 
 
class LLMRateLimitError(LLMError):
    """Rate limit exhausted after retries -> 429."""
 
 
class LLMTimeoutError(LLMError):
    """Timed out after a retry -> 504."""
 
 
class LLMConfigError(LLMError):
    """Invalid API key / auth -> 500 configuration error."""
 
 
class LLMContentBlockedError(LLMError):
    """Safety filter blocked the content -> 422."""
 
 
class LLMParseError(LLMError):
    """Could not parse JSON even after retry + regex fallback."""
 
 
# ---------------------------------------------------------------------------
# Prompt templates (DDD Section 7) — verbatim
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are an expert HR technology analyst specializing in skill assessment
and competency mapping. Your task is to analyze resume content and extract
a structured skills profile with categorization.
 
RULES:
1. ONLY identify skills that have explicit evidence in the resume text.
2. Standardize skill names to match the organization's required skills
   when the resume uses synonyms (e.g., 'JS' -> 'JavaScript',
   'k8s' -> 'Kubernetes', 'ML' -> 'Machine Learning').
3. Estimate proficiency levels (1-4) based on evidence strength:
   - Level 1 (Beginner): Mentioned in coursework, certifications,
     or 'familiar with' language. Exposure only.
   - Level 2 (Intermediate): Used in 1-2 projects or 1-2 years.
   - Level 3 (Advanced): 3+ years or multiple significant projects.
     Led implementations, solved complex problems.
   - Level 4 (Expert): 5+ years, architect-level, mentoring,
     published work, conference talks.
4. CATEGORIZE each skill using EXACTLY one of these valid categories:
   {valid_categories_json}
5. Assign a confidence score (0.0-1.0) reflecting evidence quality.
6. Include a brief evidence snippet (max 50 words) from the resume.
7. If a required skill has NO evidence, DO NOT include it.
8. You MAY identify additional skills beyond the required list.
 
Respond with ONLY a valid JSON array. No markdown, no preamble.
[
  {{
    "skill_name": "Python",
    "estimated_level": 3,
    "confidence": 0.9,
    "category": "programming_languages",
    "evidence_snippet": "5 years Python, built Django REST APIs"
  }}
]"""
 
EXTRACTION_USER_PROMPT = """## RESUME CONTENT (retrieved chunks)
{resume_context}
 
## ROLE REQUIREMENTS
Role: {role_name}
Required Skills:
{role_skills_json}
 
## VALID SKILL CATEGORIES
{valid_categories_json}
 
Analyze the resume content above. Extract all identifiable skills,
standardize names to match required skills where applicable,
estimate levels (1-4), assign a category from the valid list,
and provide evidence. Return ONLY a JSON array."""
 
RERANK_SYSTEM_PROMPT = """You are an intelligent learning advisor. Given an employee's skill gaps
and a list of candidate courses, rank and filter courses to create a
personalized learning path.
 
RULES:
1. For each gap, select the top 3-5 most relevant courses.
2. Rank by APPROPRIATENESS to employee's current level:
   - L1->L3 gap: beginner courses first, then intermediate.
   - L2->L4 gap: skip beginner, recommend intermediate + advanced.
   - Missing skill (null->Lx): start from very beginning.
3. Filter out redundant courses covering same content.
4. Consider provider quality and ratings when tie-breaking.
5. Provide brief reasoning (1-2 sentences) per recommendation.
6. Assign relevance_score (0.0-1.0) for each course.
7. Group recommendations by skill gap.
 
Respond with ONLY a valid JSON array. No markdown, no explanation.
[
  {{
    "skill_name": "Python",
    "courses": [
      {{
        "course_id": "COURSE-PL-INT-001",
        "title": "...",
        "provider": "...",
        "level": "Intermediate",
        "relevance_score": 0.92,
        "reasoning": "Bridges L2->L3 gap with focus on..."
      }}
    ]
  }}
]"""
 
RERANK_USER_PROMPT = """## EMPLOYEE SKILL GAPS
{gaps_json}
 
## CANDIDATE COURSES (retrieved via semantic search)
{candidate_courses_json}
 
## CONTEXT
Employee: {employee_name}
Target Role: {role_name}
 
Select and rank the best courses for each gap. Return ONLY JSON."""
 
STRICTER_SUFFIX = (
    "\n\nIMPORTANT: Your previous response was not valid JSON. "
    "Respond with ONLY a raw JSON array. Do not include markdown code fences, "
    "comments, or any text before or after the JSON."
)
 
 
# ---------------------------------------------------------------------------
# Low-level Gemini call with backoff
# ---------------------------------------------------------------------------
def _configure() -> None:
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "your-gemini-api-key-here":
        raise LLMConfigError("GEMINI_API_KEY is not configured")
    import google.generativeai as genai
 
    genai.configure(api_key=settings.gemini_api_key)
 
 
def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
) -> str:
    """Single Gemini generate call with 429 backoff + timeout retry."""
    _configure()
    import google.generativeai as genai
    from google.api_core import exceptions as gexc
 
    settings = get_settings()
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system_prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
        },
    )
 
    logger.info("Calling Gemini model '%s' (temp=%s)", settings.gemini_model, temperature)
    attempt = 0
    timed_out_once = False
    while True:
        try:
            response = model.generate_content(
                user_prompt, request_options={"timeout": timeout}
            )
            return _extract_response_text(response)
        except gexc.ResourceExhausted as exc:
            if attempt >= MAX_BACKOFF_RETRIES:
                raise LLMRateLimitError("LLM rate limit reached") from exc
            delay = BACKOFF_SCHEDULE[min(attempt, len(BACKOFF_SCHEDULE) - 1)]
            logger.warning("Gemini 429; backing off %ss (attempt %d)", delay, attempt + 1)
            time.sleep(delay)
            attempt += 1
        except (gexc.DeadlineExceeded, TimeoutError) as exc:
            if timed_out_once:
                raise LLMTimeoutError("LLM service timeout") from exc
            timed_out_once = True
            logger.warning("Gemini timeout; retrying once")
        except (gexc.Unauthenticated, gexc.PermissionDenied) as exc:
            logger.error("Gemini auth/permission error: %r", exc)
            raise LLMConfigError(f"Gemini auth error: {exc}") from exc
        except (gexc.InvalidArgument, gexc.NotFound) as exc:
            # Bad model name, unsupported generation_config, etc.
            logger.error("Gemini invalid request (model='%s'): %r", settings.gemini_model, exc)
            raise LLMConfigError(f"Gemini invalid request: {exc}") from exc
        except LLMError:
            raise
        except Exception as exc:  # surface anything else with detail
            logger.exception("Unexpected Gemini call failure")
            raise LLMError(f"Gemini call failed: {exc}") from exc
 
 
def _extract_response_text(response: Any) -> str:
    """Extract text, raising typed errors on safety blocks / empty output."""
    # Prompt-level block.
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise LLMContentBlockedError("Resume flagged by safety filter")
 
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        # finish_reason == 3 (SAFETY) in the protobuf enum.
        if finish_reason == 3:
            raise LLMContentBlockedError("Resume flagged by safety filter")
 
    try:
        text = response.text or ""
    except Exception:  # pragma: no cover - blocked/empty candidate access
        text = ""
    return text.strip()
 
 
# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()
 
 
def _coerce_to_list(parsed: Any) -> Optional[List[Any]]:
    """Coerce a parsed JSON value into a list of records.
 
    Accepts a top-level array, an object wrapping an array (e.g.
    {"skills": [...]}), or a single record object (wrapped into a list).
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # An object whose value is the array we want.
        for value in parsed.values():
            if isinstance(value, list):
                return value
        # A single record object -> wrap it.
        if parsed:
            return [parsed]
    return None
 
 
def _regex_extract_json_array(text: str) -> Optional[Any]:
    """Last-resort: grab the outermost [...] block and try to parse it."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
 
 
def _repair_truncated_array(text: str) -> Optional[List[Any]]:
    """Repair a JSON array that was cut off mid-stream (MAX_TOKENS).
 
    Finds the array start, keeps every complete top-level object, and closes
    the array. Returns whatever complete records were recovered.
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    last_complete = -1  # index just after the last balanced top-level object
    for i in range(start + 1, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete = i + 1
    if last_complete == -1:
        return None
    candidate = text[start:last_complete] + "]"
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None
 
 
def _parse_json_array(text: str) -> Optional[List[Any]]:
    if not text:
        return None
    cleaned = _strip_code_fences(text)
 
    # 1. Direct parse (accept arrays, wrapped arrays, or single objects).
    try:
        coerced = _coerce_to_list(json.loads(cleaned))
        if coerced is not None:
            return coerced
    except json.JSONDecodeError:
        pass
 
    # 2. Regex-extract an embedded array.
    extracted = _regex_extract_json_array(cleaned)
    if isinstance(extracted, list):
        return extracted
 
    # 3. Repair a truncated array (recover complete records).
    repaired = _repair_truncated_array(cleaned)
    if repaired is not None:
        logger.warning("Recovered %d records from a truncated JSON array", len(repaired))
        return repaired
 
    return None
 
 
# ---------------------------------------------------------------------------
# Public API - LLM Call 1: skill extraction
# ---------------------------------------------------------------------------
def extract_skills(
    resume_context: str,
    role_name: str,
    role_skills_json: str,
) -> List[Dict[str, Any]]:
    """Call 1: extract, categorize and level skills from resume context."""
    valid_categories_json = json.dumps(VALID_CATEGORIES)
    system = EXTRACTION_SYSTEM_PROMPT.format(valid_categories_json=valid_categories_json)
    user = EXTRACTION_USER_PROMPT.format(
        resume_context=resume_context,
        role_name=role_name,
        role_skills_json=role_skills_json,
        valid_categories_json=valid_categories_json,
    )
 
    text = _call_gemini(system, user, temperature=0.1, max_output_tokens=8192, timeout=30)
    parsed = _parse_json_array(text)
 
    if parsed is None:
        # Retry once with a stricter instruction (DDD 6 Stage 7).
        logger.warning(
            "Extraction JSON parse failed; retrying with stricter prompt. Raw (first 800 chars): %s",
            (text or "")[:800],
        )
        text = _call_gemini(
            system, user + STRICTER_SUFFIX, temperature=0.0, max_output_tokens=8192, timeout=30
        )
        parsed = _parse_json_array(text)
 
    if parsed is None:
        # Empty response edge case -> return empty list rather than crash (DDD 12.2).
        logger.error(
            "Extraction returned unparseable output; returning empty skill list. Raw (first 800 chars): %s",
            (text or "")[:800],
        )
        return []
 
    return _sanitize_extracted_skills(parsed)
 
 
def _sanitize_extracted_skills(raw: List[Any]) -> List[Dict[str, Any]]:
    """Validate/normalize extracted skills; fallback category to soft_skills."""
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("skill_name", "")).strip()
        if not name:
            continue
        try:
            level = int(item.get("estimated_level", 1))
        except (TypeError, ValueError):
            level = 1
        level = max(1, min(4, level))
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        category = item.get("category", FALLBACK_CATEGORY)
        if category not in VALID_CATEGORIES:
            logger.warning("Invalid category '%s' for '%s'; fallback to %s", category, name, FALLBACK_CATEGORY)
            category = FALLBACK_CATEGORY
        out.append(
            {
                "skill_name": name,
                "estimated_level": level,
                "confidence": round(confidence, 2),
                "category": category,
                "evidence_snippet": str(item.get("evidence_snippet", ""))[:400],
            }
        )
    return out
 
 
# ---------------------------------------------------------------------------
# Public API - LLM Call 2: course re-ranking
# ---------------------------------------------------------------------------
def rerank_courses(
    gaps_json: str,
    candidate_courses_json: str,
    employee_name: str,
    role_name: str,
) -> List[Dict[str, Any]]:
    """Call 2: rank/filter candidate courses into a personalized path."""
    user = RERANK_USER_PROMPT.format(
        gaps_json=gaps_json,
        candidate_courses_json=candidate_courses_json,
        employee_name=employee_name or "N/A",
        role_name=role_name,
    )
 
    text = _call_gemini(
        RERANK_SYSTEM_PROMPT, user, temperature=0.2, max_output_tokens=4096, timeout=45
    )
    parsed = _parse_json_array(text)
 
    if parsed is None:
        logger.warning("Re-ranking JSON parse failed; retrying with stricter prompt")
        text = _call_gemini(
            RERANK_SYSTEM_PROMPT,
            user + STRICTER_SUFFIX,
            temperature=0.1,
            max_output_tokens=4096,
            timeout=45,
        )
        parsed = _parse_json_array(text)
 
    if parsed is None:
        logger.error("Re-ranking returned unparseable output; returning empty recommendations")
        return []
 
    return [item for item in parsed if isinstance(item, dict)]