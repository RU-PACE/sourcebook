"""Backward-compatible storage helper facade.

Existing application modules still import ``app.s3_client``. The provider-aware
implementation lives in ``app.storage.s3_compatible`` and targets MinIO by default.
"""
from app.storage.s3_compatible import (  # noqa: F401
    copy_object,
    create_download_url,
    create_presigned_upload_post,
    delete_object,
    download_object_bytes,
    download_object_to_path,
    ensure_bucket,
    get_presign_client,
    get_s3_client,
    head_object,
    upload_bytes,
)

__all__ = [
    "copy_object", "create_download_url", "create_presigned_upload_post", "delete_object",
    "download_object_bytes", "download_object_to_path", "ensure_bucket", "get_presign_client",
    "get_s3_client", "head_object", "upload_bytes",
]
