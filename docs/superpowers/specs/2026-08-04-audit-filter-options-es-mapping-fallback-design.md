# Audit Filter Options Elasticsearch Mapping Fallback

## Context

The audit-center page loads records through `/v1/audit/search` and loads its
searchable dropdown values through `/v1/audit/filter-options`. The latter runs
`terms` aggregations over `user`, `api_key_name`, `model_id`, `model_name`, and
`client_ip` in the `xinference-audit-*` indices.

The repository's audit index template maps these fields directly as `keyword`.
An index created without that template can instead dynamically map them as
`text` fields with a `.keyword` subfield. A normal audit search can still
succeed, while aggregating the base `text` field fails with Elasticsearch's
`fielddata is disabled` error. The API currently turns that response into an
unqualified HTTP 502.

The supplied Elasticsearch log also shows a separate startup interval in which
the `.apm-source-map` primary shard was unavailable and the cluster was red.
That shard later started and the cluster became yellow. It does not identify the
audit filter request and therefore is not sufficient evidence that the audit
failure is a shard-availability problem.

## Goals

- Keep the existing audit API and response schema unchanged.
- Support audit indices whose dropdown fields are direct `keyword` fields or
  dynamically-created `text` fields with `.keyword` subfields.
- Retry only the known mapping compatibility failure.
- Preserve HTTP 502 for genuine Elasticsearch failures.

## Non-goals

- Rebuild or migrate Elasticsearch indices.
- Hide authentication, timeout, unavailable-shard, or arbitrary query errors.
- Change audit search matching or frontend behavior.
- Support a wildcard target containing a mixture of both mapping layouts in a
  single request. Operators should keep concrete audit indices aligned with the
  repository template; this fallback covers deployments whose selected indices
  consistently use either layout.

## Design

`list_audit_filter_options` will retain its current first request, aggregating
the five base field names. The Elasticsearch request will be factored so the
same session, URL, time range, authentication, and timeout behavior can issue a
second aggregation body when needed.

If the first response is successful, the handler returns its buckets exactly as
it does today. If Elasticsearch returns HTTP 400 and its error body identifies
the `fielddata is disabled` condition, the handler retries once with each
aggregation field changed from `<field>` to `<field>.keyword`. The aggregation
keys remain the original API field names, so the JSON response remains stable.

No retry occurs for any other status or error body. A failed retry is logged
with its Elasticsearch status and response excerpt, then returned as the
existing HTTP 502 `Elasticsearch query failed`. Connection errors and timeouts
retain the existing `Audit service unavailable` response.

## Error handling

The mapping-error check is deliberately narrow: it requires HTTP 400 plus the
Elasticsearch `fielddata is disabled` message. This prevents permission errors,
missing or unavailable shards, malformed requests, and service failures from
being misclassified as mapping compatibility issues.

Only one retry is allowed. Both attempts use the existing ten-second request
timeout; there is no general retry loop.

## Testing

Focused async unit tests will cover:

1. A successful base-field aggregation performs one request and preserves the
   existing request body and response.
2. A base-field `fielddata is disabled` response retries with `.keyword` fields
   and returns the retry's buckets under the existing response keys.
3. An unrelated Elasticsearch error does not retry and returns HTTP 502.
4. A failed `.keyword` retry returns HTTP 502 and does not attempt again.

The focused `xinference/api/tests/test_admin.py` tests will be run after the
implementation. Broader model, GPU, and frontend suites are unrelated to this
localized backend change.
