"""Offline tests for storage selection and the frontend/static asset structure."""
from pathlib import Path
from unittest.mock import patch

from app.core import config
from app.storage import s3_compatible


ROOT = Path(__file__).resolve().parents[1]


def test_minio_is_the_default_local_storage_provider():
    # Provider and endpoint defaults are non-secret; credentials are supplied in .env.
    assert config.STORAGE_PROVIDER in {"minio", "s3"}
    if config.STORAGE_PROVIDER == "minio":
        assert config.S3_BUCKET_NAME
        assert config.MINIO_ENDPOINT.startswith(("http://", "https://"))
        assert config.MINIO_PUBLIC_ENDPOINT.startswith(("http://", "https://"))


def test_presigned_client_uses_public_minio_endpoint():
    s3_compatible.get_presign_client.cache_clear()
    try:
        with (
            patch.object(s3_compatible, "MINIO_ACCESS_KEY", "test-access-key"),
            patch.object(s3_compatible, "MINIO_SECRET_KEY", "test-secret-key"),
            patch.object(s3_compatible.boto3, "client") as client_factory,
        ):
            client_factory.return_value = object()
            s3_compatible.get_presign_client()
            kwargs = client_factory.call_args.kwargs
            if config.STORAGE_PROVIDER == "minio":
                assert kwargs["endpoint_url"] == config.MINIO_PUBLIC_ENDPOINT
    finally:
        s3_compatible.get_presign_client.cache_clear()


def test_frontend_assets_are_split_and_referenced():
    html = (ROOT / "frontend" / "index.html").read_text()
    assert './assets/css/main.css' in html
    assert './assets/js/app.js' in html
    assert '<script type="module"' in html
    for module in ("api.js", "auth.js", "workspaces.js", "documents.js", "ingestion.js", "search.js"):
        assert (ROOT / "frontend" / "assets" / "js" / module).is_file()


def test_backend_compatibility_import_paths_are_kept():
    assert (ROOT / "app" / "config.py").is_file()
    assert (ROOT / "app" / "db_client.py").is_file()
    assert (ROOT / "app" / "s3_client.py").is_file()


def test_migration_file_sha256_is_streaming_and_repeatable():
    from io import BytesIO
    from scripts.migrate_s3_to_minio import file_sha256

    payload = BytesIO(b"sourcebook\x00object-bytes")
    first = file_sha256(payload)
    second = file_sha256(payload)
    assert first == second
    assert len(first) == 64
    assert payload.tell() == 0


def test_migration_matching_prefers_sha256_metadata_when_present():
    from scripts.migrate_s3_to_minio import object_matches

    source = {"ContentLength": 7, "ETag": '"source-etag"', "Metadata": {"sha256": "abc"}}
    matching = {"ContentLength": 7, "ETag": '"other-etag"', "Metadata": {"sha256": "abc"}}
    different = {"ContentLength": 7, "ETag": '"source-etag"', "Metadata": {"sha256": "def"}}
    assert object_matches(source, matching)
    assert not object_matches(source, different)
