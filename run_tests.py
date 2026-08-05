import subprocess
import sys


def main() -> None:
    """Discover and run the complete test suite with pytest."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-v",
        ],
        check=False,
    )

    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()