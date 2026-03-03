import argparse
import subprocess
import sys
from pathlib import Path


def run_all_ops_tests(device: str, profile: bool) -> None:
    ops_dir = Path(__file__).resolve().parent / "ops"

    test_files = sorted(
        f for f in ops_dir.glob("*.py") if f.name != "__init__.py"
    )

    if not test_files:
        print("No ops test files found.")
        return

    print(f"Running {len(test_files)} ops tests on {device}...")

    failed = []
    for test_file in test_files:
        cmd = [sys.executable, str(test_file), "--device", device]
        if profile:
            cmd.append("--profile")

        print(f"\n>>> Running {test_file.name}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            failed.append((test_file.name, result.returncode))

    if failed:
        print("\n\033[91mSome ops tests failed:\033[0m")
        for name, code in failed:
            print(f" - {name} (exit code: {code})")
        sys.exit(1)

    print("\n\033[92mAll ops tests passed!\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    run_all_ops_tests(device=args.device, profile=args.profile)
