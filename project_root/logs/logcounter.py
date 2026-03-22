import re

log_file = "preprocessing_log.txt"

# Counters
total_files = 0
success_files = 0
skipped_files = 0
error_files = 0
total_chunks = 0

# Patterns
success_pattern = re.compile(r"SUCCESS: .* → (\d+) chunks")
skipped_pattern = re.compile(r"SKIPPED")
error_pattern = re.compile(r"ERROR|FAIL", re.IGNORECASE)

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        total_files += 1

        # SUCCESS case
        success_match = success_pattern.search(line)
        if success_match:
            success_files += 1
            total_chunks += int(success_match.group(1))
            continue

        # SKIPPED case
        if skipped_pattern.search(line):
            skipped_files += 1
            continue

        # Explicit ERROR case
        if error_pattern.search(line):
            error_files += 1
            continue

# Implicit errors (missing classification)
classified = success_files + skipped_files + error_files
implicit_errors = total_files - classified

# Final error count
total_errors = error_files + implicit_errors

# Output
print("===== LOG SUMMARY =====")
print(f"Total files processed : {total_files}")
print(f"Successful files      : {success_files}")
print(f"Skipped files         : {skipped_files}")
print(f"Error files (explicit): {error_files}")
print(f"Error files (implicit): {implicit_errors}")
print(f"Total errors          : {total_errors}")
print(f"Total chunks created  : {total_chunks}")

# Consistency check
if success_files + skipped_files == total_files:
    print("\n✔ CONSISTENCY CHECK PASSED")
else:
    print("\n✘ CONSISTENCY CHECK FAILED")
    print(f"Missing files: {total_files - (success_files + skipped_files)}")