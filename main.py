"""Terminal entry point for the version-preserving experiment runner."""

import sys

from experiment_runner import JournalExperimentRunner


def main(argv=None):
    try:
        return JournalExperimentRunner.run_cli(argv)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Experiment runner: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Experiment interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
