import json
import random
import time
from pathlib import Path
from sih2026.extraction.pdf import extract_pdf
from sih2026.processing.cleaner import clean_document
from sih2026.nlp_processing import run_pipeline

print("=== RUNNING FULL PIPELINE WITH DISCOVERY AND ITERATOR FIXES ===", flush=True)

pdf_p = Path("../input/sylabus.pdf")
doc = extract_pdf(pdf_p)
cleaned_doc = clean_document(doc)
st1 = cleaned_doc.model_dump()

t0 = time.time()
res = run_pipeline(st1, verbose=True)
t_elapsed = time.time() - t0

out_p = Path("data/output/sylabus_candidates.json")
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

skills = res["candidate_skills"]

print(f"\n[PIPELINE RESULTS]")
print(f"Time taken: {t_elapsed:.2f}s")
print(f"Raw candidates extracted: {res['stats']['raw_candidates']}")
print(f"Original final_candidates: 7573")
print(f"Previous final_candidates: 3286")
print(f"NEW final_candidates count: {len(skills)}")
print(f"Total reduction from original: {7573 - len(skills)} ({(7573 - len(skills))/7573*100:.1f}% reduction!)")

# Specific noise check for previously identified bugs
specific_checks = [
    "Terrence W Pratt", "Jean Paul", "Cheng Liu", "Sangeeta Sharma", "Michael McCarthy",
    "Rajib Mall", "Complete Reference", "Select and apply", "Apply sorting", "Develop",
    "Apply the concepts", "LEXICAL ANALYZER Lexical", "INTRODUCTION Introduction",
    "Projects", "data", "comm", "semantics"
]
found_checks = [s['text'] for s in skills if s['text'] in specific_checks]
print(f"\nSpecific noise examples check (All must be GONE []): {found_checks}")

# Pull fresh 40 random sample entries
random.seed(42)
sample_40 = random.sample(skills, min(40, len(skills)))

print("\n--- FRESH 40 RANDOM SAMPLE CANDIDATES ---")
for idx, s in enumerate(sample_40, 1):
    print(f"{idx:2d}. text='{s['text']}' | category={s.get('category')} | methods={s['methods']} | occurrences={s['occurrence_count']}")
