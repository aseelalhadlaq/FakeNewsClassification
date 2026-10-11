# ============================================================
# SECTION 12: Final summary
# ============================================================

print()
print("=" * 70)
print("FULL RUN COMPLETE. All outputs saved under:", OUTPUT_ROOT)
print("=" * 70)
for f in sorted(OUTPUT_ROOT.rglob("*.csv")):
    print(" ", f.relative_to(OUTPUT_ROOT))