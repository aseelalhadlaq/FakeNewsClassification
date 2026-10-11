from pathlib import Path
import zipfile

working_dir = Path("/kaggle/working")
output_zip = working_dir / "kaggle_working_backup.zip"

with zipfile.ZipFile(
    output_zip,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
) as zip_file:

    for path in working_dir.rglob("*"):
        if not path.is_file():
            continue

        # Prevent the ZIP file from adding itself.
        if path.resolve() == output_zip.resolve():
            continue

        archive_name = path.relative_to(working_dir)
        zip_file.write(path, arcname=archive_name)

print(f"ZIP created successfully: {output_zip}")
print(f"Size: {output_zip.stat().st_size / (1024**2):.2f} MB")