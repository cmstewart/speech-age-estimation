"""Select the 250 MyST students for Studies 2A and 2B.

This is the selection code as it ran on May 7, 2026. It was recovered from
the session where it ran and is kept verbatim below, apart from the input
path and the omitted summary printing.

Input is a plain-text listing of every file path in the MyST corpus
LDC2021S05, one path per line. A student is eligible with at least 10
utterances that have both a .flac file and a .trn file. The script draws
250 eligible students with random.sample under seed 42 and writes
study2_manifest.json.

The candidate order follows the line order of the file listing. Seed 42
reproduces the same 250 students only with the same listing in the same
order. The manifest holds corpus student ids and file paths, so it is not
distributed with this repository.

Usage
    python select_study2_students.py myst_full_file_list.txt
"""

import re
import sys
import json
import random
from collections import defaultdict

with open(sys.argv[1]) as f:
    lines = f.readlines()

# Parse all paired utterances per student
file_pattern = re.compile(
    r'myst_child_conv_speech/data/(train|development|test)/(\d{6})/'
    r'(myst_\d{6}_[\d\-]+_[\d\-]+_([A-Z]{2,3})_[\d\.]+)/'
    r'(myst_\d{6}_[\d\-]+_[\d\-]+_([A-Z]{2,3})_[\d\.]+_(\d+))\.(flac|trn)'
)

# For each student, collect paired utterances with full paths
student_utterances = defaultdict(lambda: defaultdict(dict))
student_modules = defaultdict(set)
student_partitions = {}

for line in lines:
    line = line.strip()
    if '._' in line:
        continue
    match = file_pattern.search(line)
    if match:
        partition = match.group(1)
        student = match.group(2)
        session = match.group(3)
        module = match.group(4)
        utterance_id = match.group(5)
        utt_num = match.group(7)
        ext = match.group(8)

        # Store path relative to corpus root
        path = line.split('myst_child_conv_speech/')[-1]
        student_utterances[student][utterance_id][ext] = path
        student_modules[student].add(module)
        student_partitions[student] = partition

# Find students with >= 10 paired utterances
viable_students = {}
for student, utterances in student_utterances.items():
    paired = {uid: paths for uid, paths in utterances.items()
              if 'flac' in paths and 'trn' in paths}
    if len(paired) >= 10:
        viable_students[student] = {
            'paired_count': len(paired),
            'modules': sorted(student_modules[student]),
            'partition': student_partitions[student],
            'utterances': paired,
        }

print(f"Viable students (>= 10 paired utterances): {len(viable_students)}")

# Random sample of 250
random.seed(42)
selected_ids = sorted(random.sample(list(viable_students.keys()), 250))

print(f"Selected: {len(selected_ids)}")

# Build manifest
manifest = []
for student in selected_ids:
    info = viable_students[student]

    # Sort utterances chronologically by utterance ID (which contains date/time)
    sorted_utts = sorted(info['utterances'].keys())

    utt_list = []
    for uid in sorted_utts:
        paths = info['utterances'][uid]
        utt_list.append({
            'utterance_id': uid,
            'flac_path': paths['flac'],
            'trn_path': paths['trn'],
        })

    manifest.append({
        'student_id': student,
        'partition': info['partition'],
        'modules': info['modules'],
        'total_paired_utterances': info['paired_count'],
        'utterances': utt_list,
    })

with open('study2_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
