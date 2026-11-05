import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()


def download_tfpbd(
    workspace: str | None = None,
    project: str | None = None,
    version: int | None = None,
    export_format: str = "yolov8",
    local_cache: str = "outputs/data/tfpbd",
) -> Path:
    workspace = workspace or os.getenv("ROBOFLOW_WORKSPACE")
    project = project or os.getenv("ROBOFLOW_PROJECT")
    version = version or int(os.getenv("ROBOFLOW_VERSION", "6"))

    if not os.getenv("ROBOFLOW_API_KEY"):
        raise ValueError("Set ROBOFLOW_API_KEY in .env")

    cache_dir = Path(local_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    rf_project = rf.workspace(workspace).project(project)
    rf_version = rf_project.version(version)

    dataset = rf_version.download(
        model_format=export_format,
        location=str(cache_dir),
        overwrite=False,
    )

    dataset_path = Path(dataset.location).resolve()

    if not (dataset_path / "data.yaml").exists():
        candidates = list(cache_dir.rglob("data.yaml"))
        if not candidates:
            raise FileNotFoundError(
                f"Download finished, but no data.yaml was found under {cache_dir}. "
                "Check the Roboflow export format and API response."
            )
        dataset_path = candidates[0].parent.resolve()

    path_file = Path("outputs/data/dataset_path.txt")
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_file.write_text(str(dataset_path), encoding="utf-8")

    print(f"Dataset root: {dataset_path}")
    print(f"Saved dataset path to: {path_file}")
    return dataset_path


if __name__ == "__main__":
    download_tfpbd()