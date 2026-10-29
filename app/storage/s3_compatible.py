"""S3-compatible private object storage used by Sourcebook.

The default provider is local MinIO. The boto3 S3 API and existing object-key
layout are retained to preserve document identity and lifecycle behavior. Set
STORAGE_PROVIDER=s3 to use AWS S3 instead.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from functools import lru_cache
from urllib.parse import quote

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import (
    AWS_DEFAULT_REGION,
    CORS_ORIGINS,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_PUBLIC_ENDPOINT,
    MINIO_SECRET_KEY,
    S3_BUCKET_NAME,
    S3_ENDPOINT_URL,
    STORAGE_PROVIDER,
)

logger = logging.getLogger(__name__)


def _build_client(*, endpoint_url: str | None, for_presigned_url: bool = False):
    if STORAGE_PROVIDER not in {"minio", "s3"}:
        raise RuntimeError("STORAGE_PROVIDER must be either 'minio' or 's3'.")

    if STORAGE_PROVIDER == "minio":
        access_key = MINIO_ACCESS_KEY
        secret_key = MINIO_SECRET_KEY
        if not access_key or not secret_key:
            raise RuntimeError("Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY in .env.")
        session_token = None
    else:
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")
        if bool(access_key) != bool(secret_key):
            raise RuntimeError(
                "Set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or use an IAM role."
            )

    options = {
        "region_name": AWS_DEFAULT_REGION,
        "endpoint_url": endpoint_url,
        "config": Config(
            signature_version="s3v4",
            max_pool_connections=50,
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    }
    if access_key and secret_key:
        options["aws_access_key_id"] = access_key
        options["aws_secret_access_key"] = secret_key
        if session_token:
            options["aws_session_token"] = session_token
    return boto3.client("s3", **options)


@lru_cache(maxsize=1)
def get_s3_client():
    """Return the SDK client used for server-to-storage operations."""
    endpoint = MINIO_ENDPOINT if STORAGE_PROVIDER == "minio" else S3_ENDPOINT_URL
    return _build_client(endpoint_url=endpoint)


@lru_cache(maxsize=1)
def get_presign_client():
    """Return the client used for links consumed by the browser.

    A Dockerized API may reach MinIO at ``http://minio:9000`` but a host browser
    cannot resolve the Docker service name. MINIO_PUBLIC_ENDPOINT deliberately
    controls the host embedded in the presigned signature.
    """
    endpoint = MINIO_PUBLIC_ENDPOINT if STORAGE_PROVIDER == "minio" else S3_ENDPOINT_URL
    return _build_client(endpoint_url=endpoint, for_presigned_url=True)


def _require_bucket() -> str:
    bucket = S3_BUCKET_NAME.strip()
    if not bucket:
        raise RuntimeError("The object-storage bucket is not configured.")
    return bucket


def ensure_bucket() -> bool:
    """Ensure the local MinIO bucket and browser CORS policy exist.

    Returns ``True`` if the bucket was created. AWS buckets remain operator-managed;
    this helper never creates or changes a bucket when STORAGE_PROVIDER=s3.
    """
    if STORAGE_PROVIDER != "minio":
        return False
    bucket = _require_bucket()
    client = get_s3_client()
    created = False
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        response = exc.response or {}
        code = str((response.get("Error") or {}).get("Code", ""))
        status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode", 0))
        if code not in {"404", "NoSuchBucket", "NotFound"} and status != 404:
            raise RuntimeError(
                f"Could not verify MinIO bucket '{bucket}'. Check MINIO_ENDPOINT and credentials."
            ) from exc
        try:
            client.create_bucket(Bucket=bucket)
            created = True
        except ClientError as create_exc:
            create_response = create_exc.response or {}
            create_code = str((create_response.get("Error") or {}).get("Code", ""))
            if create_code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise RuntimeError(f"Could not create MinIO bucket '{bucket}'.") from create_exc

    # Some MinIO Community releases do not implement bucket-level PutBucketCors.
    # Those releases apply CORS globally; keep startup working and make the limitation visible.
    try:
        client.put_bucket_cors(
            Bucket=bucket,
            CORSConfiguration={
                "CORSRules": [
                    {
                        "AllowedOrigins": CORS_ORIGINS,
                        "AllowedMethods": ["POST", "GET", "HEAD"],
                        "AllowedHeaders": ["*"],
                        "ExposeHeaders": ["ETag", "x-amz-checksum-sha256", "x-amz-meta-document-id", "x-amz-meta-sha256"],
                        "MaxAgeSeconds": 3000,
                    }
                ]
            },
        )
    except ClientError as exc:
        response = exc.response or {}
        code = str((response.get("Error") or {}).get("Code", ""))
        http_status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode", 0))
        if code.lower() not in {"notimplemented", "not implemented"} and http_status != 501:
            raise RuntimeError(f"Could not configure CORS for MinIO bucket '{bucket}'.") from exc
        logger.warning(
            "Bucket-level CORS is unsupported by this MinIO build; its global CORS policy will be used. "
            "Configured frontend origins: %s",
            ", ".join(CORS_ORIGINS),
        )
    return created


def create_presigned_upload_post(
    staging_key: str,
    content_type: str,
    document_id: str,
    sha256_hex: str,
    max_file_size_bytes: int,
    expires_in: int = 900,
) -> dict:
    """Create a short-lived POST policy for one staging key and expected checksum."""
    bucket = _require_bucket()
    checksum_b64 = base64.b64encode(bytes.fromhex(sha256_hex)).decode("ascii")
    fields = {
        "Content-Type": content_type,
        "x-amz-checksum-algorithm": "SHA256",
        "x-amz-checksum-sha256": checksum_b64,
        "x-amz-meta-document-id": document_id,
        "x-amz-meta-sha256": sha256_hex,
    }
    conditions = [
        ["content-length-range", 1, int(max_file_size_bytes)],
        {"Content-Type": content_type},
        {"x-amz-checksum-algorithm": "SHA256"},
        {"x-amz-checksum-sha256": checksum_b64},
        {"x-amz-meta-document-id": document_id},
        {"x-amz-meta-sha256": sha256_hex},
    ]
    return get_presign_client().generate_presigned_post(
        Bucket=bucket,
        Key=staging_key,
        Fields=fields,
        Conditions=conditions,
        ExpiresIn=int(expires_in),
    )


def head_object(s3_key: str) -> dict:
    """Read object metadata and checksum when the selected S3-compatible server exposes it."""
    bucket = _require_bucket()
    try:
        return get_s3_client().head_object(Bucket=bucket, Key=s3_key, ChecksumMode="ENABLED")
    except ClientError as exc:
        code = str((exc.response.get("Error") or {}).get("Code", ""))
        if code not in {"InvalidArgument", "NotImplemented", "501"}:
            raise
        return get_s3_client().head_object(Bucket=bucket, Key=s3_key)


def copy_object(source_key: str, destination_key: str) -> None:
    """Promote a validated staging object to its immutable document-version key."""
    bucket = _require_bucket()
    get_s3_client().copy_object(
        Bucket=bucket,
        Key=destination_key,
        CopySource={"Bucket": bucket, "Key": source_key},
        MetadataDirective="COPY",
    )


def delete_object(s3_key: str) -> None:
    """Delete one object. This is intentionally scoped to an exact object key."""
    get_s3_client().delete_object(Bucket=_require_bucket(), Key=s3_key)


def create_download_url(s3_key: str, filename: str, expires_in: int = 300) -> str:
    """Create a short-lived URL for downloading one private object."""
    bucket = _require_bucket()
    safe_filename = quote(filename.replace("\r", "").replace("\n", ""), safe="")
    return get_presign_client().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": s3_key,
            "ResponseContentDisposition": f"attachment; filename*=UTF-8''{safe_filename}",
        },
        ExpiresIn=int(expires_in),
    )


def upload_bytes(
    data: bytes,
    s3_key: str,
    content_type: str,
    metadata: dict[str, str] | None = None,
) -> str:
    """Write a private artifact and return its S3-compatible URI."""
    bucket = _require_bucket()
    safe_metadata = {str(k).lower(): str(v)[:1024] for k, v in (metadata or {}).items()}
    safe_metadata.setdefault("sha256", hashlib.sha256(data).hexdigest())
    params = {
        "Bucket": bucket,
        "Key": s3_key,
        "Body": data,
        "ContentType": content_type,
        "Metadata": safe_metadata,
    }
    # Local MinIO deployments normally do not have server-side encryption configured.
    # Preserve the previous AWS S3 SSE-S3 default only for the actual AWS provider.
    sse = (
        os.getenv("S3_SERVER_SIDE_ENCRYPTION", "AES256").strip()
        if STORAGE_PROVIDER == "s3"
        else os.getenv("MINIO_SERVER_SIDE_ENCRYPTION", "").strip()
    )
    if sse:
        params["ServerSideEncryption"] = sse
    get_s3_client().put_object(**params)
    return f"s3://{bucket}/{s3_key}"


def download_object_to_path(s3_key: str, destination_path: str) -> None:
    """Download a private object to a local path for extraction."""
    get_s3_client().download_file(_require_bucket(), s3_key, destination_path)


def download_object_bytes(s3_key: str, bucket: str | None = None) -> bytes:
    """Download a small derived artifact; never used for the original upload byte stream."""
    target_bucket = (bucket or _require_bucket()).strip()
    if not target_bucket:
        raise RuntimeError("The object-storage bucket is not configured.")
    response = get_s3_client().get_object(Bucket=target_bucket, Key=s3_key)
    body = response.get("Body")
    if body is None:
        raise RuntimeError("S3 object response did not contain a body")
    try:
        return body.read()
    finally:
        body.close()


__all__ = [
    "get_s3_client", "get_presign_client", "ensure_bucket",
    "create_presigned_upload_post", "head_object", "copy_object", "delete_object",
    "create_download_url", "upload_bytes", "download_object_to_path", "download_object_bytes",
]
