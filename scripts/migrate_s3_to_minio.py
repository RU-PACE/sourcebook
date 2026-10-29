#!/usr/bin/env python3
"""Non-destructive, opt-in copy of one S3 bucket into the configured MinIO bucket.

Dry-run is the default. Nothing is deleted. Existing destination objects are skipped
only when size and available ETag/SHA-256 metadata match; conflicts stop the run unless
--overwrite is explicitly supplied with --execute. Copied objects are streamed through a
temporary file and verified with a computed SHA-256 digest before the migration continues.

Source credentials use AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN.
Destination credentials use MINIO_ACCESS_KEY/MINIO_SECRET_KEY and MINIO_ENDPOINT.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def make_client(*, endpoint_url: str | None, region: str, access_key: str | None,
                secret_key: str | None, session_token: str | None = None):
    options: dict[str, Any] = {
        "region_name": region,
        "endpoint_url": endpoint_url,
        "config": Config(
            signature_version="s3v4",
            max_pool_connections=10,
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


def file_sha256(fileobj: Any) -> str:
    """Hash a seekable file object without loading the full object into memory."""
    digest = hashlib.sha256()
    fileobj.seek(0)
    while True:
        chunk = fileobj.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    fileobj.seek(0)
    return digest.hexdigest()


def object_matches(source_head: dict[str, Any], destination_head: dict[str, Any]) -> bool:
    if int(source_head.get("ContentLength", -1)) != int(destination_head.get("ContentLength", -2)):
        return False
    source_meta = {str(k).lower(): str(v) for k, v in (source_head.get("Metadata") or {}).items()}
    destination_meta = {str(k).lower(): str(v) for k, v in (destination_head.get("Metadata") or {}).items()}
    source_sha = source_meta.get("sha256")
    destination_sha = destination_meta.get("sha256")
    if source_sha and destination_sha:
        return source_sha == destination_sha
    source_etag = str(source_head.get("ETag", "")).strip('"')
    destination_etag = str(destination_head.get("ETag", "")).strip('"')
    return bool(source_etag and destination_etag and source_etag == destination_etag)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bucket", required=True, help="Existing AWS S3/source bucket name")
    parser.add_argument("--destination-bucket", default=os.getenv("MINIO_BUCKET", "sourcebook"))
    parser.add_argument("--prefix", default="", help="Optional object-key prefix to copy")
    parser.add_argument("--execute", action="store_true", help="Actually copy objects; without this flag, report only")
    parser.add_argument("--overwrite", action="store_true", help="Replace conflicting destination objects (requires --execute)")
    args = parser.parse_args()

    if args.overwrite and not args.execute:
        parser.error("--overwrite requires --execute")

    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"
    source = make_client(
        endpoint_url=os.getenv("S3_SOURCE_ENDPOINT_URL") or None,
        region=region,
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        session_token=os.getenv("AWS_SESSION_TOKEN"),
    )
    destination = make_client(
        endpoint_url=(os.getenv("MINIO_ENDPOINT") or "http://localhost:9000").rstrip("/"),
        region=region,
        access_key=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "sourcebook")),
        secret_key=os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD"),
    )

    destination.head_bucket(Bucket=args.destination_bucket)
    paginator = source.get_paginator("list_objects_v2")
    seen = copied = skipped = 0
    conflicts: list[str] = []
    mode = "COPY MODE" if args.execute else "DRY RUN — no objects will be written"
    print(f"{mode}: s3://{args.source_bucket}/{args.prefix} -> s3://{args.destination_bucket}/")

    for page in paginator.paginate(Bucket=args.source_bucket, Prefix=args.prefix):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            seen += 1
            try:
                source_head = source.head_object(Bucket=args.source_bucket, Key=key)
            except ClientError as exc:
                print(f"ERROR: unable to inspect source object {key!r}: {exc}", file=sys.stderr)
                return 2
            try:
                destination_head = destination.head_object(Bucket=args.destination_bucket, Key=key)
            except ClientError as exc:
                error = (exc.response.get("Error") or {}).get("Code", "")
                status = (exc.response.get("ResponseMetadata") or {}).get("HTTPStatusCode", 0)
                if error not in {"404", "NoSuchKey", "NotFound"} and status != 404:
                    print(f"ERROR: unable to inspect destination object {key!r}: {exc}", file=sys.stderr)
                    return 2
                destination_head = None

            if destination_head and object_matches(source_head, destination_head):
                skipped += 1
                print(f"SKIP (already matches): {key}")
                continue
            if destination_head and not args.overwrite:
                conflicts.append(key)
                print(f"CONFLICT (destination differs): {key}", file=sys.stderr)
                continue
            print(f"{'COPY' if args.execute else 'WOULD COPY'} ({item.get('Size', 0)} bytes): {key}")
            if not args.execute:
                continue

            try:
                # Disk-backed streaming avoids holding large source documents in memory.
                # A computed digest is stored as object metadata and then checked against
                # the bytes downloaded back from MinIO, rather than trusting ETag alone.
                with tempfile.TemporaryFile() as source_file:
                    source.download_fileobj(args.source_bucket, key, source_file)
                    digest = file_sha256(source_file)
                    source_head = dict(source_head)
                    metadata = {
                        str(k).lower(): str(v)
                        for k, v in (source_head.get("Metadata") or {}).items()
                    }
                    metadata["sha256"] = digest
                    extra_args: dict[str, Any] = {"Metadata": metadata}
                    for header, arg in (
                        ("ContentType", "ContentType"),
                        ("CacheControl", "CacheControl"),
                        ("ContentDisposition", "ContentDisposition"),
                        ("ContentEncoding", "ContentEncoding"),
                        ("ContentLanguage", "ContentLanguage"),
                    ):
                        if source_head.get(header):
                            extra_args[arg] = source_head[header]

                    destination.upload_fileobj(
                        source_file,
                        args.destination_bucket,
                        key,
                        ExtraArgs=extra_args,
                    )

                verified_head = destination.head_object(Bucket=args.destination_bucket, Key=key)
                if int(verified_head.get("ContentLength", -1)) != int(source_head.get("ContentLength", -2)):
                    raise RuntimeError("Destination object size does not match source")
                if (verified_head.get("Metadata") or {}).get("sha256") != digest:
                    raise RuntimeError("Destination SHA-256 metadata does not match source bytes")
                with tempfile.TemporaryFile() as destination_file:
                    destination.download_fileobj(args.destination_bucket, key, destination_file)
                    destination_digest = file_sha256(destination_file)
                if destination_digest != digest:
                    raise RuntimeError("Destination object bytes failed SHA-256 verification")
                copied += 1
            except Exception as exc:
                print(f"ERROR: copy or verification failed for {key!r}: {exc}", file=sys.stderr)
                return 2

    print(f"\nSummary: discovered={seen}, copied={copied}, already_matching={skipped}, conflicts={len(conflicts)}")
    if conflicts:
        print("Conflicts were not overwritten. Review them, then rerun with --overwrite --execute only if intended.")
        return 1
    if not args.execute:
        print("No data changed. Re-run with --execute to copy after reviewing this plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
