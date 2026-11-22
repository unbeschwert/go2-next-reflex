"""Regression tests for dataset archives consumed by the existing trainer."""

import csv
import io
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from .layout import archive_people, check_person_splits

HERE = Path(__file__).resolve().parent
SAMPLE = struct.pack(">IHHBq", 13, 120, 140, 1, 100)


class DatasetToolsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.clips = self.root / "clips"
        self.captures = self.root / "captures"

    def add_clip(self, session, name):
        folder = self.clips / session
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(SAMPLE)

    def set_person(self, session, person):
        folder = self.captures / f"s{session}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "person.txt").write_text(person + "\n")

    def command(self, script, *args):
        return subprocess.run([sys.executable, "-B", str(HERE / script), *args],
                              capture_output=True, text=True)

    def export(self, sessions, *args):
        return self.command("export_sessions.py", "--sessions", sessions, "--dataset", str(self.clips),
                            "--captures", str(self.captures), "--out-dir", str(self.root), *args)

    def test_person_aliases_and_legacy_class_are_preserved(self):
        self.add_clip("20", "gesture_01-heart-normal-01.bin")
        self.add_clip("20", "gesture_04-namaskaram-normal-01.bin")
        self.add_clip("23", "gesture_04-follow-me-front-01.bin")
        self.add_clip("28", "gesture_05-no-action-normal-01.bin")
        self.add_clip("29", "gesture_05-no-action-normal-01.bin")
        self.set_person("22", "20")
        self.set_person("23", "22")
        sessions = "20,23,28,29"
        invalid = self.export(sessions, "--val", "23", "--test", "29")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("person(s) in more than one split", invalid.stderr)
        valid = self.export(sessions, "--val", "28", "--test", "29")
        self.assertEqual(valid.returncode, 0, valid.stderr)
        with zipfile.ZipFile(self.root / "go2_gestures_s20-23-28-29.zip") as archive:
            self.assertFalse(any("namaskaram" in name for name in archive.namelist()))
            rows = list(csv.DictReader(io.StringIO(archive.read("labels.csv").decode())))
            follow, = [row for row in rows if row["class_id"] == "4"]
            self.assertEqual((follow["class_name"], follow["label_index"], follow["person"]),
                             ("follow-me", "3", "20"))
            self.assertIn("--class-names", archive.read("README.md").decode())
            people = archive_people(archive)
            self.assertEqual(people["20"], people["23"])
        with self.assertRaisesRegex(ValueError, r"person\(s\) in more than one split"):
            check_person_splits(people, {"train": ["20"], "test": ["23"]})

    def test_time_split_uses_real_folders_and_current_training_command(self):
        for i in range(30):
            self.add_clip("20", f"gesture_04-follow-me-front-{i + 1:02d}.bin")
        result = self.export("20", "--time-split", "70,15,15")
        self.assertEqual(result.returncode, 0, result.stderr)
        with zipfile.ZipFile(self.root / "go2_gestures_s20_timesplit.zip") as archive:
            # This mode deliberately evaluates different clips from the same person.
            self.assertEqual(archive_people(archive), {})
            self.assertIn("--val 120 --test 220", archive.read("README.md").decode())
            self.assertIn("train_snn_v8.py", archive.read("README.md").decode())

    def test_packer_requires_explicit_remapping_of_conflicting_labels(self):
        self.add_clip("20", "gesture_01-heart-normal-01.bin")
        self.add_clip("20", "gesture_04-follow-me-front-01.bin")
        base = self.root / "base.zip"
        output = self.root / "packed.zip"
        with zipfile.ZipFile(base, "w") as archive:
            archive.writestr("motion_buddy_dataset/01/gesture_01-sit-normal-01.bin", SAMPLE)
            archive.writestr("motion_buddy_dataset/01/gesture_05-no-action-normal-01.bin", SAMPLE)
        args = ("--base", str(base), "--dataset", str(self.clips), "--sessions", "20", "--out", str(output))
        refused = self.command("pack_dataset.py", *args)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("conflicts with base class", refused.stderr)
        self.assertFalse(output.exists())
        result = self.command("pack_dataset.py", *args, "--as-no-action")
        self.assertEqual(result.returncode, 0, result.stderr)
        with zipfile.ZipFile(output) as archive:
            originals = [name for name in archive.namelist() if name.startswith("motion_buddy_dataset/01/")]
            new = [name for name in archive.namelist() if name.startswith("motion_buddy_dataset/20/")]
            self.assertEqual(len(originals), 2)
            self.assertEqual(len(new), 2)
            self.assertTrue(all("gesture_05-no-action-" in name for name in new))
            self.assertEqual(archive.read(new[0]), SAMPLE)

    def test_labeler_dry_run_and_move_produce_valid_clip_name(self):
        samples = self.root / "samples"
        samples.mkdir()
        sample = samples / "sample_1.bin"
        sample.write_bytes(SAMPLE)
        args = ("--samples", str(samples), "--dataset", str(self.clips), "--session", "20",
                "--cls", "4", "--label", "follow-me", "--style", "front")
        dry = self.command("label_recordings.py", *args, "--dry-run")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertTrue(sample.is_file())
        result = self.command("label_recordings.py", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.clips / "20" / "gesture_04-follow-me-front-01.bin").read_bytes(), SAMPLE)
        self.assertFalse(sample.exists())
