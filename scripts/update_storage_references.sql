-- Explicit metadata retargeting after you have copied and verified every referenced object.
-- Run only after backing up PostgreSQL and confirming the copy destination bucket name.
-- Example:
--   psql "$DATABASE_URL" -v source_bucket='legacy-s3-bucket' -v target_bucket='sourcebook' -f scripts/update_storage_references.sql

BEGIN;

UPDATE documents
SET storage_uri = 's3://' || :'target_bucket' || '/' || s3_key
WHERE storage_uri LIKE ('s3://' || :'source_bucket' || '/%');

UPDATE extracted_artifacts
SET s3_bucket = :'target_bucket'
WHERE s3_bucket = :'source_bucket';

COMMIT;
