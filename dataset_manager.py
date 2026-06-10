"""
Dataset management abstraction for Azure Blob Storage.

Supports uploading, downloading, listing, and retrieving the latest
dataset version. Works with both local files and Azure Blob.

Environment variables:
    AZURE_STORAGE_CONNECTION_STRING  — Azure Blob connection string
    AZURE_STORAGE_ACCOUNT           — storage account name
    AZURE_STORAGE_CONTAINER         — container name (default: datasets)
"""
import os
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

DATASET_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "datasets")
LATEST_BLOB_NAME = "latest_water_potability.csv"


class DatasetManager:
    """Manages dataset versioning in Azure Blob Storage.

    Datasets are stored as:
        datasets/<version_timestamp>/water_potability.csv
        datasets/latest_water_potability.csv   (alias)

    Falls back to local filesystem when Azure credentials are not available.
    """

    def __init__(self):
        self._blob_service_client = None
        self._container_client = None
        self._use_azure = False
        self._init_azure()

    def _init_azure(self):
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
        if conn_str:
            try:
                from azure.storage.blob import BlobServiceClient
                self._blob_service_client = BlobServiceClient.from_connection_string(conn_str)
                self._container_client = self._blob_service_client.get_container_client(
                    DATASET_CONTAINER
                )
                self._use_azure = True
                print(f"DatasetManager: connected to Azure Blob '{DATASET_CONTAINER}'")
            except Exception as e:
                print(f"DatasetManager: failed to connect to Azure Blob ({e}), using local fs")
        elif account_name:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.storage.blob import BlobServiceClient
                credential = DefaultAzureCredential()
                account_url = f"https://{account_name}.blob.core.windows.net"
                self._blob_service_client = BlobServiceClient(account_url, credential=credential)
                self._container_client = self._blob_service_client.get_container_client(
                    DATASET_CONTAINER
                )
                self._use_azure = True
                print(f"DatasetManager: connected to Azure Blob '{DATASET_CONTAINER}' via managed identity")
            except Exception as e:
                print(f"DatasetManager: failed to connect via managed identity ({e}), using local fs")
        else:
            print("DatasetManager: no Azure credentials found, using local filesystem")

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def upload_dataset(self, local_path: str, version: Optional[str] = None) -> str:
        """Upload a dataset to Azure Blob with versioning.

        Args:
            local_path: Path to local CSV file.
            version: Version string. Auto-generated if None.

        Returns:
            The version string used.
        """
        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local dataset not found: {local_path}")

        if self._use_azure:
            blob_name = f"{version}/water_potability.csv"
            with open(local_path, "rb") as data:
                blob_client = self._container_client.get_blob_client(blob_name)
                blob_client.upload_blob(data, overwrite=True)
                print(f"Uploaded: {blob_name}")

            latest_blob = self._container_client.get_blob_client(LATEST_BLOB_NAME)
            with open(local_path, "rb") as data:
                latest_blob.upload_blob(data, overwrite=True)
                print(f"Uploaded: {LATEST_BLOB_NAME}")
        else:
            print(f"Local mode: dataset at {local_path} (version={version})")

        return version

    def download_dataset(self, version: str, dest_dir: str = "dataset") -> str:
        """Download a specific dataset version from Azure Blob.

        Args:
            version: Version string or 'latest'.
            dest_dir: Local directory to save the file.

        Returns:
            Path to the downloaded file.
        """
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, "water_potability.csv")

        if self._use_azure:
            blob_name = f"{version}/water_potability.csv" if version != "latest" else LATEST_BLOB_NAME
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(dest_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            print(f"Downloaded: {blob_name} → {dest_path}")
        else:
            if not os.path.exists(dest_path):
                raise FileNotFoundError(
                    f"No dataset found locally at {dest_path}. "
                    "Set AZURE_STORAGE_CONNECTION_STRING to download from Blob."
                )
            print(f"Local mode: using {dest_path}")

        return dest_path

    def list_versions(self) -> List[str]:
        """List all dataset version folders in Azure Blob."""
        if not self._use_azure:
            print("Local mode: no version history available")
            return []

        versions = set()
        blobs = self._container_client.list_blobs()
        for blob in blobs:
            match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{6})/", blob.name)
            if match:
                versions.add(match.group(1))

        return sorted(versions, reverse=True)

    def get_latest_version(self) -> Optional[str]:
        """Get the most recent dataset version string."""
        versions = self.list_versions()
        return versions[0] if versions else None

    def download_latest(self, dest_dir: str = "dataset") -> Tuple[str, str]:
        """Download the latest dataset version.

        Returns:
            (local_path, version_string)
        """
        version = self.get_latest_version() or "latest"
        path = self.download_dataset(version, dest_dir)
        return path, version

    @property
    def is_azure_connected(self) -> bool:
        return self._use_azure
