#!/usr/bin/env python
"""
One-off migration: copy every experiment/run and the model registry from the
current MLflow file store (DATA_ROOT/mlruns) into a new SQLite-backed store
(DATA_ROOT/mlflow.db) -- what MLflow's own deprecation warnings ask for
("The filesystem tracking backend ... is deprecated ... consider ...
'sqlite:///mlflow.db'").

Only METADATA moves (params, metrics with full history, tags, run status/
timestamps, registered model versions, aliases). Artifacts (the actual model
files, plots, etc.) are left exactly where they are -- this only changes
where params/metrics/registry ENTRIES are recorded, not where the files
they point to live. That also means this does NOT fix the separate,
pre-existing issue where those files are recorded under a Docker-container-
absolute path (file:///data/mlruns/...) and can't be loaded directly from a
different host -- see src/Inference/README.md's "Known gaps" section.

Registered model versions in this MLflow install use the 3.x "Logged Model"
source scheme (source="models:/m-<id>"), which -- confirmed by testing --
requires that Logged Model to actually exist in whatever store you're
registering against. Rather than also recreating Logged Model entities
(a materially bigger, newer MLflow concept with its own params/metrics that
would just duplicate what the parent run already has), each version is
re-registered with `source` rewritten to the Logged Model's own real
artifact_location (a plain file:// path, confirmed NOT to require any
registry lookup) -- same underlying files, same portability characteristics,
just resolved directly instead of through an intermediate registry entry
this script isn't recreating.

Safe to run: only reads the existing file store and writes a brand-new
sqlite file. Doesn't touch or delete anything under mlruns/. Re-runnable
if MLFLOW_SQLITE_PATH doesn't already exist; refuses to overwrite one that
does (see main() below) rather than silently duplicating every run.

Usage:
    python scripts/migrate_mlflow_to_sqlite.py
"""
import os
import sys
import yaml

from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.tracking import MlflowClient

DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", r"a:\Dominic\My Projects\Home Credit\data")
OLD_MLRUNS_DIR = os.path.join(DATA_ROOT, "mlruns")
SQLITE_PATH = os.path.join(DATA_ROOT, "mlflow.db")

OLD_TRACKING_URI = "file:" + OLD_MLRUNS_DIR.replace("\\", "/")
NEW_TRACKING_URI = "sqlite:///" + SQLITE_PATH.replace("\\", "/")


def resolve_logged_model_artifact_path(experiment_dir: str, model_id: str) -> str:
    """
    A registered version's source is "models:/m-<id>" -- a reference to a
    Logged Model entity, not a direct path. Reads that Logged Model's own
    meta.yaml (DATA_ROOT/mlruns/<experiment_id>/models/<model_id>/meta.yaml)
    for its real artifact_location instead, so the migrated version doesn't
    depend on a Logged Model row existing in the new store.
    """
    meta_path = os.path.join(experiment_dir, "models", model_id, "meta.yaml")
    with open(meta_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    return meta["artifact_location"]


def migrate_runs(src: MlflowClient, dst: MlflowClient) -> dict:
    """Returns {old_run_id: new_run_id} for every migrated run."""
    run_id_map = {}
    experiments = [e for e in src.search_experiments(view_type=ViewType.ALL) if e.experiment_id != "0"]

    for exp in experiments:
        print(f"\n=== Experiment: {exp.name} ({exp.experiment_id}) ===")
        dst_exp = dst.get_experiment_by_name(exp.name)
        if dst_exp is None:
            # Keep using the same shared artifact folder going forward --
            # this migration only ever touches the metadata store.
            new_exp_id = dst.create_experiment(exp.name, artifact_location=f"file:{OLD_MLRUNS_DIR}".replace("\\", "/"))
            print(f"Created destination experiment id={new_exp_id}")
        else:
            new_exp_id = dst_exp.experiment_id
            print(f"Reusing existing destination experiment id={new_exp_id}")

        runs = src.search_runs([exp.experiment_id], run_view_type=ViewType.ALL, max_results=5000)
        runs = sorted(runs, key=lambda r: r.info.start_time or 0)  # oldest first, for readable ordering
        print(f"{len(runs)} run(s) to migrate")

        for old_run in runs:
            tags = {k: v for k, v in old_run.data.tags.items() if k != "mlflow.runName"}
            tags["mlflow.migrated_from_run_id"] = old_run.info.run_id
            tags["mlflow.migrated_from_tracking_uri"] = OLD_TRACKING_URI

            new_run = dst.create_run(
                experiment_id=new_exp_id,
                start_time=old_run.info.start_time,
                run_name=old_run.info.run_name,
                tags=tags,
            )
            new_run_id = new_run.info.run_id

            params = [Param(k, v) for k, v in old_run.data.params.items()]
            metrics = []
            for key in old_run.data.metrics:
                metrics.extend(
                    Metric(m.key, m.value, m.timestamp, m.step)
                    for m in src.get_metric_history(old_run.info.run_id, key)
                )
            dst.log_batch(new_run_id, metrics=metrics, params=params)
            dst.set_terminated(new_run_id, status=old_run.info.status, end_time=old_run.info.end_time)

            run_id_map[old_run.info.run_id] = new_run_id
            print(f"  {old_run.info.run_id} -> {new_run_id}  ({old_run.info.run_name}, "
                  f"{len(params)} params, {len(old_run.data.metrics)} metric keys)")

    return run_id_map


def migrate_registry(src: MlflowClient, dst: MlflowClient, run_id_map: dict) -> None:
    models = src.search_registered_models()
    print(f"\n=== {len(models)} registered model(s) ===")

    for model in models:
        try:
            dst.get_registered_model(model.name)
            print(f"Reusing existing registered model {model.name!r}")
        except Exception:
            dst.create_registered_model(model.name, description=model.description)
            print(f"Created registered model {model.name!r}")

        versions = sorted(src.search_model_versions(f"name='{model.name}'"), key=lambda v: int(v.version))
        version_map = {}  # old version number (str) -> new ModelVersion

        for v in versions:
            if v.source.startswith("models:/"):
                model_id = v.source.split("models:/")[-1]
                if not v.run_id:
                    raise ValueError(
                        f"{model.name} v{v.version} has a models:/ source but no run_id -- "
                        "can't resolve which experiment its Logged Model directory lives under."
                    )
                experiment_id = src.get_run(v.run_id).info.experiment_id
                experiment_dir = os.path.join(OLD_MLRUNS_DIR, experiment_id)
                real_source = resolve_logged_model_artifact_path(experiment_dir, model_id)
            else:
                real_source = v.source  # already a plain path (older-style registration)

            new_run_id = run_id_map.get(v.run_id)
            new_version = dst.create_model_version(
                name=model.name,
                source=real_source,
                run_id=new_run_id,
                tags=v.tags,
                description=v.description,
            )
            # search_model_versions() returns .version as an int, but
            # RegisteredModel.aliases below maps alias -> version as a str --
            # normalize to str so the aliases loop's lookup actually hits.
            version_map[str(v.version)] = new_version
            print(f"  v{v.version} -> v{new_version.version}  source={real_source}")

        for alias, version in (model.aliases or {}).items():
            dst.set_registered_model_alias(model.name, alias, version_map[str(version)].version)
            print(f"  alias {alias!r} -> v{version_map[str(version)].version}")


def main():
    if os.path.exists(SQLITE_PATH):
        print(f"Refusing to run: {SQLITE_PATH} already exists.")
        print("Delete it first if you really want to re-migrate from scratch.")
        sys.exit(1)

    print(f"Source (file store) : {OLD_TRACKING_URI}")
    print(f"Destination (sqlite) : {NEW_TRACKING_URI}")

    src = MlflowClient(tracking_uri=OLD_TRACKING_URI)
    dst = MlflowClient(tracking_uri=NEW_TRACKING_URI)

    run_id_map = migrate_runs(src, dst)
    migrate_registry(src, dst, run_id_map)

    print(f"\nDone. {len(run_id_map)} run(s) migrated to {SQLITE_PATH}")
    print("Set MLFLOW_TRACKING_URI to the sqlite URI above in docker-compose.yml, "
          "the training notebooks, and src/Inference/config.py before relying on this.")


if __name__ == "__main__":
    main()
