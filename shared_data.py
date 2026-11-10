"""Validate cleaning, build dataset-wide IDs and reproduce exact indexed splits."""
import argparse
import csv
import hashlib
import io
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DRUG, PROTEIN = "drug_id", "protein_id"
SMILES, SEQUENCE = "compound_iso_smiles", "target_sequence"
PAIR_KEYS = [DRUG, PROTEIN]
TEXT_KEYS = [SMILES, SEQUENCE]
RAW_COLUMNS = TEXT_KEYS + ["affinity"]
INDEXED_COLUMNS = PAIR_KEYS + RAW_COLUMNS
SCHEMA = "dataset_wide_sorted_identity_v1"
CLEANED = {"davis": "task2/data/davis_clean_task2.csv", "kiba": "data/kiba_clean.csv"}
EXPECTED = {"davis": (68, 379, 25772), "kiba": (2068, 229, 117657)}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids(frame, column):
    if column not in frame:
        raise ValueError(f"Missing canonical ID column: {column}")
    values = frame[column]
    if (values.isna().any() or pd.api.types.is_bool_dtype(values.dtype)
            or not pd.api.types.is_integer_dtype(values.dtype)
            or ((values < 0) | (values > np.iinfo(np.int64).max)).any()):
        raise ValueError(f"{column} must contain nonnegative int64 identifiers")
    return values.to_numpy(dtype=np.int64)


def axis(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu" or not np.array_equal(values, np.arange(len(values))):
        raise ValueError(f"Invalid canonical {name} axis")
    return values


def positions(frame, column, count, fallback=False):
    values = ids(frame, column)
    unknown = values >= count
    if unknown.any() and not fallback:
        raise ValueError(f"Unknown canonical IDs in {column}: {int(unknown.sum())} rows")
    return np.where(unknown, -1, values) if fallback else values


def training_axes(training, n_drugs=None, n_proteins=None):
    """Require full warm-start coverage of the declared canonical axes."""
    result = []
    for column, given, attribute in ((DRUG, n_drugs, "n_drugs"), (PROTEIN, n_proteins, "n_proteins")):
        values = ids(training, column)
        count = given if given is not None else training.attrs.get(attribute)
        if count is None:
            count = int(values.max()) + 1 if len(values) else 0
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or count <= 0 or not np.array_equal(np.unique(values), np.arange(count))):
            raise ValueError(f"Task 2 training must cover the full contiguous canonical {column} axis")
        result.append(np.arange(count, dtype=np.int64))
    return tuple(result)


def mapping_metadata(training):
    return {key: np.asarray(training.attrs[key]) for key in
            ("canonical_schema", "drug_mapping_sha256", "protein_mapping_sha256", "drug_strings", "protein_strings")
            if key in training.attrs}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = csv.reader(stream)
        if next(rows, None) != RAW_COLUMNS:
            raise ValueError(f"Unexpected source columns: {path}")
        for row in rows:
            if len(row) != 3 or not all(value.strip() for value in row[:2]) or not math.isfinite(float(row[2])):
                raise ValueError(f"Invalid observation in {path}; missing labels are never zero-filled")
            yield row


def canonical_strings(cleaned):
    """Only identity columns participate: labels cannot influence ID assignment."""
    return sorted(set(cleaned[SMILES])), sorted(set(cleaned[SEQUENCE]))


def validate_cleaning(dataset, root=ROOT):
    """Audit raw mean aggregation; preserve validated cleaned decimal cells."""
    root = Path(root)
    raw = pd.read_csv(root / "data" / f"{dataset}_all.csv")
    raw["affinity"] = pd.to_numeric(raw.affinity, errors="coerce")
    finite = np.isfinite(raw.affinity.to_numpy(dtype=float))
    valid_ids = raw[TEXT_KEYS].notna().all(axis=1)
    for column in TEXT_KEYS:
        raw[column] = raw[column].astype("string").str.strip()
        valid_ids &= raw[column].ne("")
    valid = raw.loc[finite & valid_ids, RAW_COLUMNS].copy()
    grouped = valid.groupby(TEXT_KEYS, as_index=False, sort=False).affinity.mean()
    known = pd.read_csv(root / CLEANED[dataset], float_precision="round_trip")
    a, b = (frame.set_index(TEXT_KEYS).sort_index() for frame in (grouped, known))
    if not a.index.equals(b.index) or known.duplicated(TEXT_KEYS).any():
        raise ValueError(f"{dataset}: cleaning changed pair identity or uniqueness")
    delta = np.abs(a.affinity.to_numpy() - b.affinity.to_numpy())
    if not np.allclose(a.affinity, b.affinity, atol=1e-12, rtol=0):
        raise ValueError(f"{dataset}: re-cleaning disagrees with validated affinities")
    actual = (known[SMILES].nunique(), known[SEQUENCE].nunique(), len(known))
    if actual != EXPECTED[dataset]:
        raise ValueError(f"{dataset}: expected counts {EXPECTED[dataset]}, observed {actual}")
    return known, dict(dataset=dataset, raw_rows=len(raw), invalid_scores=int((~finite).sum()),
                       invalid_ids=int((~valid_ids).sum()), duplicate_excess=len(valid)-len(grouped),
                       drugs=actual[0], proteins=actual[1], cleaned_rows=actual[2],
                       score_scale="supplied pKd" if dataset == "davis" else "supplied transformed KIBA",
                       score_transformation="none", cleaned_reference=CLEANED[dataset],
                       reclean_max_abs_difference=float(delta.max()), reclean_nonidentical_floats=int(np.count_nonzero(delta)),
                       cleaned_serialization_policy="retain exact validated cleaned cells after mean-aggregation audit")


def _write_verified(path, render, check_only):
    path = Path(path)
    if check_only and not path.is_file():
        raise FileNotFoundError(path)
    if not check_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        generated = Path(temp) / path.name
        with generated.open("w", newline="", encoding="utf-8") as stream:
            render(csv.writer(stream, lineterminator="\n"))
        digest = sha(generated)
        if path.exists():
            if sha(path) != digest:
                raise ValueError(f"Generated file differs; refusing to overwrite {path}")
        elif not check_only:
            with path.open("xb") as output, generated.open("rb") as source:
                for block in iter(lambda: source.read(1024*1024), b""):
                    output.write(block)
        return digest


def split_rows(dataset, cleaned_rows, root=ROOT):
    """Recover original row order and decimal tokens from compact assignments."""
    parts = {name: [] for name in ("train", "val", "test")}
    path = Path(root) / "task2/data" / f"{dataset}_split_assignment.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader) != ["split", "cleaned_row", "affinity_override"]:
            raise ValueError(f"Invalid split assignment header: {path}")
        seen = set()
        for part, index, override in reader:
            index = int(index)
            if part not in parts or index in seen or not 0 <= index < len(cleaned_rows):
                raise ValueError(f"Invalid or duplicate split assignment: {path}")
            row = cleaned_rows[index].copy()
            if override:
                row[2] = override
            if not math.isfinite(float(row[2])):
                raise ValueError("Split affinity must be finite")
            parts[part].append(row)
            seen.add(index)
    if len(seen) != len(cleaned_rows):
        raise ValueError("Split assignments must cover every cleaned observation")
    return parts


def original_csv_sha(rows):
    digest, buffer = hashlib.sha256(), io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(RAW_COLUMNS)
    for row in rows:
        writer.writerow(row)
        if buffer.tell() >= 1024*1024:
            digest.update(buffer.getvalue().encode("utf-8"))
            buffer.seek(0)
            buffer.truncate(0)
    digest.update(buffer.getvalue().encode("utf-8"))
    return digest.hexdigest()


def prepare(root=ROOT, check_only=False):
    root = Path(root)
    report = {"schema": SCHEMA, "cleaning": [], "datasets": [], "task2_splits": [],
              "id_assignment_uses_affinities": False, "test_evaluation_performed": False}
    for dataset in CLEANED:
        cleaned, cleaning = validate_cleaning(dataset, root)
        report["cleaning"].append(cleaning)
        drugs, proteins = canonical_strings(cleaned)
        dm, pm = {s:i for i,s in enumerate(drugs)}, {s:i for i,s in enumerate(proteins)}
        for kind, column, text, strings in (("drug", DRUG, SMILES, drugs), ("protein", PROTEIN, SEQUENCE, proteins)):
            def render_map(writer, column=column, text=text, strings=strings):
                writer.writerow([column, text])
                writer.writerows(enumerate(strings))
            _write_verified(root / "data" / f"{dataset}_{kind}_mapping.csv", render_map, check_only)
        cleaned_rows = list(read_rows(root / CLEANED[dataset]))
        cleaned_pairs = {tuple(row[:2]): row[2] for row in cleaned_rows}
        if len(cleaned_pairs) != len(cleaned_rows):
            raise ValueError("Duplicate cleaned pair")

        def indexed_writer(rows):
            def render(writer):
                writer.writerow(INDEXED_COLUMNS)
                for smiles, sequence, label in rows:
                    writer.writerow([dm[smiles], pm[sequence], smiles, sequence, label])
            return render

        shared = root / "data" / f"{dataset}_indexed.csv"
        shared_hash = _write_verified(shared, indexed_writer(cleaned_rows), check_only)
        with shared.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            assert next(reader) == INDEXED_COLUMNS
            for raw, row in zip(cleaned_rows, reader, strict=True):
                assert row[2:] == raw and int(row[0]) == dm[raw[0]] and int(row[1]) == pm[raw[1]]
        seen, train_drugs, train_proteins = set(), set(), set()
        for split, rows in split_rows(dataset, cleaned_rows, root).items():
            pairs = {tuple(row[:2]) for row in rows}
            if len(pairs) != len(rows) or seen.intersection(pairs) or not pairs <= cleaned_pairs.keys():
                raise ValueError(f"{dataset}/{split}: duplicate, overlap or pair absent from cleaned data")
            split_drugs, split_proteins = {r[0] for r in rows}, {r[1] for r in rows}
            if split == "train":
                train_drugs, train_proteins = split_drugs, split_proteins
            elif not split_drugs <= train_drugs or not split_proteins <= train_proteins:
                raise ValueError(f"{dataset}/{split}: warm-start coverage failed")
            differences = [abs(float(r[2]) - float(cleaned_pairs[tuple(r[:2])])) for r in rows]
            if max(differences) > 1e-12:
                raise ValueError("Existing split affinities disagree meaningfully with cleaned observations")
            target = root / "task2/data/splits" / f"{dataset}_{split}.csv"
            digest = _write_verified(target, indexed_writer(rows), check_only)
            with target.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                assert next(reader) == INDEXED_COLUMNS
                for raw, row in zip(rows, reader, strict=True):
                    assert row[2:] == raw and int(row[0]) == dm[raw[0]] and int(row[1]) == pm[raw[1]]
            seen.update(pairs)
            report["task2_splits"].append(dict(dataset=dataset, split=split, rows=len(rows),
                original_sha256=original_csv_sha(rows), indexed_sha256=digest, exact_original_cells=True,
                pairs_disjoint=True, warm_start=True,
                cleaned_vs_split_float_differences=int(np.count_nonzero(differences)),
                cleaned_vs_split_max_abs_difference=max(differences)))
        assert seen == cleaned_pairs.keys()
        report["datasets"].append(dict(dataset=dataset, rows=len(cleaned_rows), drugs=len(drugs), proteins=len(proteins),
            cleaned_sha256=sha(root/CLEANED[dataset]), indexed_sha256=shared_hash,
            dataset_wide_ids_equal_old_training_order=(drugs == sorted(train_drugs) and proteins == sorted(train_proteins))))
    path = root / "data/preparation_report.json"
    if check_only:
        assert json.loads(path.read_text()) == report
    else:
        text = json.dumps(report, indent=2) + "\n"
        if path.exists() and json.loads(path.read_text()) != report:
            raise ValueError("Preparation evidence changed; inspect before replacing it")
        if not path.exists():
            path.write_text(text)
    return report


def load_mappings(dataset, root=ROOT):
    if dataset not in CLEANED:
        raise ValueError(f"Unknown dataset: {dataset}")
    result = {"canonical_schema": SCHEMA, "dataset": dataset}
    for kind, column, text in (("drug", DRUG, SMILES), ("protein", PROTEIN, SEQUENCE)):
        path = Path(root) / "data" / f"{dataset}_{kind}_mapping.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        axis(ids(frame, column), column)
        strings = frame[text].tolist()
        if strings != sorted(set(strings)) or any(not s.strip() for s in strings):
            raise ValueError(f"Invalid canonical mapping: {path}")
        result[f"n_{kind}s"] = len(frame)
        result[f"{kind}_strings"] = strings
        result[f"{kind}_mapping_sha256"] = sha(path)
    return result


def load_task2_split(dataset, split, task2_root=None):
    """Model runners must explicitly request train/val; no automatic test read."""
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    task2_root = Path(task2_root) if task2_root is not None else ROOT / "task2"
    metadata = load_mappings(dataset, task2_root.parent)
    frame = pd.read_csv(task2_root / "data/splits" / f"{dataset}_{split}.csv", float_precision="round_trip")
    for column, kind in ((DRUG, "drug"), (PROTEIN, "protein")):
        values = positions(frame, column, metadata[f"n_{kind}s"])
        trace = SMILES if kind == "drug" else SEQUENCE
        if trace in frame:
            if not np.array_equal(frame[trace].to_numpy(dtype=str), np.asarray(metadata[f"{kind}_strings"])[values]):
                raise ValueError(f"Trace string and canonical ID disagree: {trace}")
    frame.attrs.update(metadata)
    return frame


def target_disjoint_split(frame, seed=42, train_fraction=.7, val_fraction=.15):
    """Partition protein identities without labels; preserve global IDs and row order."""
    ids(frame, DRUG)
    proteins = ids(frame, PROTEIN)
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Require positive train/validation/test fractions")
    unique = np.unique(proteins)
    if len(unique) < 3:
        raise ValueError("Need at least three targets")
    shuffled = np.random.default_rng(seed).permutation(unique)
    nt = min(max(1, int(train_fraction*len(unique))), len(unique)-2)
    nv = min(max(1, int(val_fraction*len(unique))), len(unique)-nt-1)
    groups = {"train": shuffled[:nt], "val": shuffled[nt:nt+nv], "test": shuffled[nt+nv:]}
    return {name: frame.loc[np.isin(proteins, group)].copy().reset_index(drop=True) for name, group in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = prepare(args.root, args.check_only)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
