"""Shared clip naming and person metadata for the dataset tools."""

import re
import struct

CLIP_NAME = re.compile(r"gesture_(\d+)-([a-z0-9]+(?:-[a-z0-9]+)*)-([a-z]+)-(\d+)\.bin")
CLIP_MEMBER = re.compile(r"motion_buddy_dataset/(\d+)/(gesture_[^/]+\.bin)")


def session_ids(value):
    sessions = []
    for part in value.split(","):
        part = part.strip()
        if not part.isdecimal():
            raise ValueError(f"invalid session number {part!r}")
        sessions.append(f"{int(part):02d}")
    if len(set(sessions)) != len(sessions):
        raise ValueError(f"duplicate session number in {value!r}")
    return sessions


def resolve_people(captures, sessions):
    """Follow numeric aliases, including aliases outside the exported sessions."""
    people = {}
    for session in sessions:
        current = session
        seen = set()
        while current not in people:
            if current in seen:
                raise ValueError(f"cycle in person.txt aliases for session {session}")
            seen.add(current)
            path = captures / f"s{current}" / "person.txt"
            person = path.read_text().strip() if path.exists() else current
            if not person:
                raise ValueError(f"empty person identifier in {path}")
            if not person.isdecimal():
                break
            person = f"{int(person):02d}"
            if person == current:
                break
            current = person
        else:
            person = people[current]
        people.update(dict.fromkeys(seen, person))
    return {session: people[session] for session in sessions}


def recording_size(path):
    """Return the event count of a nonempty recording with a valid length header."""
    with path.open("rb") as source:
        header = source.read(4)
    length = path.stat().st_size - 4
    if len(header) == 4 and length > 0 and length % 13 == 0 and struct.unpack(">I", header)[0] == length:
        return length // 13
    return 0


def check_person_splits(people, splits):
    person_splits = {}
    for split, sessions in splits.items():
        for session in sessions:
            person_splits.setdefault(people.get(session, session), set()).add(split)
    conflicts = {person: sorted(groups) for person, groups in person_splits.items() if len(groups) > 1}
    if conflicts:
        raise ValueError(f"person(s) in more than one split: {conflicts}")


def archive_people(archive):
    """Read person IDs from exported CSVs. Time splits opt out of person isolation."""
    import csv
    import io

    try:
        labels = archive.open("labels.csv")
    except KeyError:
        return {}
    people = {}
    with io.TextIOWrapper(labels, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            member = CLIP_MEMBER.fullmatch(row.get("path") or "")
            person = row.get("person")
            if member and person and row.get("split_mode") != "time":
                session = member[1]
                if people.setdefault(session, person) != person:
                    raise ValueError(f"session {session} has inconsistent person identifiers")
    return people
