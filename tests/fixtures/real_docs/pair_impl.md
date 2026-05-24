# Tideline — Implementation Note

*Companion to the Tideline specification. Describes the current code shape as of build 0.3.4.*

This note describes how Tideline is currently implemented. It
covers most of the spec's contract, diverges from the spec on the
persistence policy, and leaves a few spec items unimplemented.

## 1. Request path

Every public HTTP and gRPC handler invokes the `RateLimiter.check`
method before any business logic runs. The method reads the
`X-Tenant-ID` header, looks up the tenant's bucket, and returns an
`Admitted` or `Refused` verdict. Control-plane endpoints use a
separate `InternalAuth` middleware and do not reach the rate
limiter.

## 2. Token bucket

The bucket is a small struct holding `capacity`, `refill_rate`, and
`tokens`. The `check` method first calls `refill()`, which adds
`(now - last_refill) * refill_rate` tokens to the bucket up to
`capacity`. If `tokens >= 1`, the verdict is `Admitted` and `tokens`
is decremented by 1. Otherwise the verdict is `Refused` and the
caller sets a `Retry-After` header equal to `ceil(1 / refill_rate)`.

## 3. Configuration

The tenant configuration is loaded from `tideline.yaml` at process
startup. There is no `SIGHUP` reload path — operators must restart
the process to pick up configuration changes. The default policy for
unconfigured tenants is `capacity = 100, refill_rate = 10`.

## 4. Persistence

The current implementation **persists** token counts to a small
SQLite file on every refusal. On restart, buckets are restored from
this file rather than reinitialized to capacity. This was introduced
to prevent restart storms during deploy windows.

## 5. Observability

The `tideline_requests_total` and `tideline_tokens_current` metrics
are wired and exported on the `/metrics` endpoint. The
`tideline_refills_total` metric is not wired in the current build;
it is tracked in the issue tracker as a follow-up.

## 6. Performance

The single-tenant fast path executes in approximately 40
microseconds on the staging hardware. We have not benchmarked the
1k-tenant fan-out scenario described in the spec.
