import json
from pathlib import Path

import yaml

from anchorgate import Benchmark, Declaration, anchor, audit, ladder, render_markdown, to_json
from anchorgate.cli import main
from conftest import make_forecast


def _write_case(obs, tmp_path: Path) -> Path:
    obs.to_netcdf(tmp_path / "obs.nc")
    for s in (1, 2):
        make_forecast(obs, "half", seed=s).to_netcdf(tmp_path / f"half_{s}.nc")
    n = obs.n_time - obs.cutoff + 1 - 6  # origins in the fixture
    decl = {
        "name": "synthetic",
        "data": {
            "observations": {"format": "netcdf", "path": "obs.nc"},
            "forecasts": [
                {"format": "netcdf", "path": "half_1.nc"},
                {"format": "netcdf", "path": "half_2.nc"},
            ],
        },
        "scoring": {"purged": {"start": n - 12, "stop": n}},
        "gate": {"metric": "msess", "threshold": 0.0, "min_seeds": 2},
        "folds": [
            {"kind": "shuffled", "name": "shuffled", "k": 5, "seed": 42},
            {"kind": "purged", "name": "purged", "out_of_sample": True},
        ],
        "conclusions": {
            "pass": "the model carries information beyond climatology",
            "fail": "it does not",
        },
    }
    p = tmp_path / "declaration.yaml"
    p.write_text(yaml.safe_dump(decl), encoding="utf-8")
    return p


def test_declaration_hash_is_the_file_hash(obs, tmp_path):
    p = _write_case(obs, tmp_path)
    d = Declaration.load(p)
    assert d.preregistered and len(d.sha256) == 64
    assert d.score_origins(100, 6).size == 12
    assert d.conclusions["pass"].startswith("the model")
    p.write_text(p.read_text(encoding="utf-8") + "\n# edited after scoring\n", encoding="utf-8")
    assert Declaration.load(p).sha256 != d.sha256


def test_cli_validate_audit_anchor(obs, tmp_path, capsys):
    p = _write_case(obs, tmp_path)
    assert main(["validate", str(p)]) == 0
    assert "contract OK" in capsys.readouterr().out
    assert main(["audit", str(p), "--out", str(tmp_path / "rep")]) == 0
    assert main(["anchor", str(p), "--out", str(tmp_path / "rep")]) == 0
    md = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
    js = json.loads((tmp_path / "rep" / "report.json").read_text(encoding="utf-8"))
    assert "## Verdicts" in md and "PASS" in md
    assert js["declaration"]["preregistered"] is True
    assert js["results"][0]["component"].startswith("C1")


def test_report_renders_without_declaration(obs):
    bench = Benchmark([make_forecast(obs, "half", seed=s) for s in (1, 2)], obs)
    results = [audit(bench, ladder(bench), out_of_sample=["purged"]), anchor(bench)]
    md = render_markdown(results, bench)
    assert "C0 fold audit" in md and "C1 anchoring" in md
    doc = json.loads(to_json(results))
    assert doc["declaration"] is None and len(doc["results"]) == 2


def _write_factorial_case(obs, tmp_path: Path) -> Path:
    obs.to_netcdf(tmp_path / "obs.nc")
    forecasts = []
    for s in (1, 2):
        for model, kind in (("half", "half"), ("noisy", "noisy")):
            name = f"{model}_{s}.nc"
            make_forecast(obs, kind, seed=s, model=model).to_netcdf(tmp_path / name)
            forecasts.append({"format": "netcdf", "path": name})
    n = obs.n_time - obs.cutoff + 1 - 6
    decl = {
        "name": "synthetic-factorial",
        "data": {
            "observations": {"format": "netcdf", "path": "obs.nc"},
            "forecasts": forecasts,
        },
        "scoring": {"purged": {"start": n - 12, "stop": n}},
        "factorial": {"n_perm": 200},
        "fusion": {
            "base": [
                {"model": "half", "features": "BASIC"},
                {"model": "noisy", "features": "BASIC"},
            ],
            "assignment": "purged",
        },
    }
    p = tmp_path / "declaration.yaml"
    p.write_text(yaml.safe_dump(decl), encoding="utf-8")
    return p


def test_cli_factorial_fusion_and_run(obs, tmp_path, capsys):
    p = _write_factorial_case(obs, tmp_path)
    assert main(["factorial", str(p), "--n-perm", "100", "--out", str(tmp_path / "f")]) == 0
    assert "C2 seed-resolved factorial" in capsys.readouterr().out
    assert main(["fusion", str(p), "--out", str(tmp_path / "g")]) == 0
    assert "C3 fusion decomposition" in capsys.readouterr().out
    assert main(["fusion", str(p), "--assignment", "shuffled"]) == 0
    assert "not out of sample" in capsys.readouterr().out
    assert main(["run", str(p), "--out", str(tmp_path / "all")]) == 0
    js = json.loads((tmp_path / "all" / "report.json").read_text(encoding="utf-8"))
    components = [r["component"][:2] for r in js["results"]]
    assert components == ["C0", "C1", "C2", "C3"]
    md = (tmp_path / "all" / "report.md").read_text(encoding="utf-8")
    assert "### contrasts" in md and "### summary" in md
