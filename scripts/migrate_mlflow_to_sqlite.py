#!/usr/bin/env python
"""
One-off migration of MLflow from the old folder-based store to SQLite.
Already done: this is what created DATA_ROOT/mlflow.db. Kept for reference;
it refuses to run again while mlflow.db exists.

What it did
-----------
Copied every experiment, run and registered model from the old file store
(DATA_ROOT/mlruns) into a new SQLite database (DATA_ROOT/mlflow.db), as
MLflow's deprecation warning for the file store recommends.

Only the records moved: parameters, metrics (with their full history),
tags, run status and times, registered model versions and aliases. The
files themselves (models, plots) stayed where they were in mlruns/. So
versions registered before the switch still point at file:///data/mlruns/...
and only load inside Docker -- see "Known gaps" in the README.

One detail: MLflow 3 registers a model version with a source like
"models:/m-<id>", which only works if a matching "Logged Model" record also
exists. Instead of recreating those records, each version was re-registered
with its source set to the model's real folder (a plain file:// path) --
the same files, just referenced directly.

Safety: it only reads the old store and writes a brand-new SQLite file;
nothing under mlruns/ is changed or deleted.

Usage:
    python scripts/migrate_mlflow_to_sqlite.py
"""
import os
import sys
import yaml

from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.tracking import MlflowClient

DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", r"a:\Dominic\My Projects\Home Credit\data")
OLD_MLRUNS_DIR = os.path.join(DATA_ROOT, "mlruns")      # old store (read only)
SQLITE_PATH = os.path.join(DATA_ROOT, "mlflow.db")      # new store (created)

OLD_TRACKING_URI = "file:" + OLD_MLRUNS_DIR.replace("\\", "/")
NEW_TRACKING_URI = "sqlite:///" + SQLITE_PATH.replace("\\", "/")


def resolve_logged_model_artifact_path(experiment_dir: str, model_id: str) -> str:
    """
    Find the real folder behind a "models:/m-<id>" source.

    Reads the model's meta.yaml in the old store
    (mlruns/<experiment_id>/models/<model_id>/meta.yaml) and returns its
    artifact_location, so the migrated version can point straight at the files.
    """
    meta_path = os.path.join(experiment_dir, "models", model_id, "meta.yaml")
    with open(meta_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    return meta["artifact_location"]


def migrate_runs(src: MlflowClient, dst: MlflowClient) -> dict:
    """
    Copy every run (except those in MLflow's empty "Default" experiment) into
    the new store. Runs get new ids there, so this returns
    {old_run_id: new_run_id} for the registry step to use.
    """
    run_id_map = {}
    experiments = [e for e in src.search_experiments(view_type=ViewType.ALL) if e.experiment_id != "0"]

    for exp in experiments:
        print(f"\n=== Experiment: {exp.name} ({exp.experiment_id}) ===")
        dst_exp = dst.get_experiment_by_name(exp.name)
        if dst_exp is None:
            # Point the new experiment at the same mlruns/ folder, so the
            # files stay where they are -- only the records move.
            new_exp_id = dst.create_experiment(exp.name, artifact_location=f"file:{OLD_MLRUNS_DIR}".replace("\\", "/"))
            print(f"Created destination experiment id={new_exp_id}")
        else:
            new_exp_id = dst_exp.experiment_id
            print(f"Reusing existing destination experiment id={new_exp_id}")

        runs = src.search_runs([exp.experiment_id], run_view_type=ViewType.ALL, max_results=5000)
        runs = sorted(runs, key=lambda r: r.info.start_time or 0)  # oldest first, for readable ordering
        print(f"{len(runs)} run(s) to migrate")

        for old_run in runs:
            # Copy the tags (the run name is set separately below) and record
            # where the run came from
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

            # Parameters, plus every logged value of every metric (not just the latest)
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
    """
    Recreate each registered model and its versions (oldest first, so the
    version numbers come out the same), linked to the migrated runs, then
    reapply aliases such as "prod".
    """
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
            # Keyed by the version as text, because the aliases below give
            # version numbers as text while v.version is a number
            version_map[str(v.version)] = new_version
            print(f"  v{v.version} -> v{new_version.version}  source={real_source}")

        for alias, version in (model.aliases or {}).items():
            dst.set_registered_model_alias(model.name, alias, version_map[str(version)].version)
            print(f"  alias {alias!r} -> v{version_map[str(version)].version}")


def main():
    # Never write into an existing database -- that would duplicate every run
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
          "the training notebooks, and src/Inference/utils/config.py before relying on this.")


if __name__ == "__main__":
    main()
