import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from firmware_performance_testing import firmware_bench as bench


class FirmwareBenchTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.output = self.root / "bench"
        self.argv = ["--setups", "nb40_own4,candidate", "--out", str(self.output)]

    def test_dry_run_rejects_different_weights_before_any_changes(self):
        files = dict.fromkeys((*bench.FIRMWARE_FILES, *bench.NETWORK_FILES), "base")
        with (patch.object(bench, "ssh") as ssh,
              patch.object(bench.subprocess, "run") as replay,
              patch.object(bench, "setup_checksums", side_effect=[files, files | {"weights.txt": "other"}]),
              patch.object(bench, "create_setups") as stage,
              patch.object(bench, "bench") as measure):
            with self.assertRaisesRegex(ValueError, r"candidate differs from nb40_own4 in weights.txt"):
                bench.main([*self.argv, "--dry-run", "--zip", "dataset.zip"])
        self.assertIn("--check-data", replay.call_args.args[0])
        ssh.assert_called_once_with("test -x /root/switch_setup.sh && test -x /root/inference.sh")
        stage.assert_not_called()
        measure.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_measurement_failure_attempts_restore_and_preserves_both_errors(self):
        files = {"nb40_own4": {}, "candidate": {}}
        with (patch.object(bench, "preflight", return_value=files),
              patch.object(bench, "create_setups"),
              patch.object(bench, "bench", side_effect=RuntimeError("replay failed")),
              patch.object(bench, "restore", side_effect=RuntimeError("camera failed")) as restore):
            with self.assertRaisesRegex(RuntimeError, "replay failed") as failure:
                bench.main(self.argv)
        restore.assert_called_once_with(files["nb40_own4"])
        self.assertEqual(failure.exception.__notes__, ["Board restoration also failed: camera failed"])

    def test_switcher_does_not_start_server_if_app_fails_to_load(self):
        setup = self.root / "setups" / "candidate"
        setup.mkdir(parents=True)
        firmware = self.root / "firmware"
        firmware.mkdir()
        (self.root / ".config" / "spleat-server").mkdir(parents=True)
        for name in (*bench.FIRMWARE_FILES, *bench.NETWORK_FILES):
            (setup / name).write_text(name)
        switcher = self.root / "switch_setup.sh"
        switcher.write_text((bench.HERE / switcher.name).read_text()
                            .replace("/usr/lib/firmware/xilinx/spleat", str(firmware))
                            .replace("/root", str(self.root)))
        self.stub(self.root / "inference.sh", "exit 0")
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        self.stub(bin_dir / "sleep", "exit 0")
        self.stub(bin_dir / "systemctl", 'printf "%s\n" "$*" >> "$CALLS"')
        self.stub(bin_dir / "xmutil", """
case "$1" in
  listapps) if [ -e "$LOADED" ]; then echo "spleat XRT_FLAT 0->0"; else echo "spleat XRT_FLAT -1"; fi ;;
  loadapp) [ "$FAIL_LOAD" != 1 ] && touch "$LOADED" ;;
esac
""")
        calls = self.root / "calls"
        env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}", "CALLS": str(calls),
                            "LOADED": str(self.root / "loaded")}
        args = ["sh", str(switcher), "candidate"]
        failed = subprocess.run(args, env=env | {"FAIL_LOAD": "1"}, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(calls.read_text().splitlines(), ["stop spleat-server"])
        self.assertNotIn("server active", failed.stdout)

        success = subprocess.run(args, env=env, capture_output=True, text=True, check=True)
        self.assertIn("setup candidate | server active | app 1", success.stdout)
        self.assertEqual((self.root / "weights.txt").read_text(), "weights.txt")

    def test_archived_runs_reproduce_published_summary(self):
        archive = bench.HERE / "results" / "2026-11-22_nb40_strategies"
        saved = json.loads((archive / "results.json").read_text())
        for run in saved["runs"]:
            run["eval"]["npz"] = archive / run["eval"]["npz"]
        with contextlib.redirect_stdout(io.StringIO()):
            actual = bench.summary(saved["runs"], list(saved["summary"]))
        self.assertEqual(actual, saved["summary"])

    @staticmethod
    def stub(path, contents):
        path.write_text(f"#!/bin/sh\n{contents}\n")
        path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
