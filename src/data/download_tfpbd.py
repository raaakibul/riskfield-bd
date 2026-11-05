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
    """Download TFP-BD directly from Roboflow at runtime.

    The user does not need to manually download or unzip the dataset.
    """
    workspace = workspace or os.getenv("ROBOFLOW_WORKSPACE")
    project = project or os.getenv("ROBOFLOW_PROJECT")
    version = version or int(os.getenv("ROBOFLOW_VERSION", "6"))

    if not os.getenv("ROBOFLOW_API_KEY"):
        raise ValueError(
            "Set ROBOFLOW_API_KEY in your .env file. "
            "Do not commit the .env file to GitHub."
        )

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    rf_workspace = rf.workspace(workspace)
    rf_project = rf_workspace.project(project)
    rf_version = rf_project.version(version)

    dataset = rf_version.download(
        model_format=export_format,
        location=local_cache,
        overwrite=False,
    )

    dataset_path = Path(dataset.location)
    print(f"Dataset available at: {dataset_path.resolve()}")
    return dataset_path


if __name__ == "__main__":
    download_tfpbd()