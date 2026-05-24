# Tideline — Rate Limiter Specification

*Version 0.3. Component specification.*

Tideline is a per-tenant rate limiter. It sits in front of every
public API endpoint and refuses requests once a tenant has exceeded
its configured budget. This document specifies the contract; see the
companion impl note for the current code shape.

## 1. Scope

Tideline applies to every HTTP and gRPC request that carries a
tenant identifier in the `X-Tenant-ID` header. Internal control-plane
requests are out of scope and must use a separate authentication
path.

## 2. Algorithm

Tideline implements a **token bucket** per tenant. Each bucket has:

- a capacity *C* expressed in requests;
- a refill rate *R* expressed in requests per second;
- a current token count *T*, initialized to *C*.

For each incoming request:

1. The bucket's token count is refilled by `(now - last_refill) * R`,
   clamped to *C*.
2. If `T >= 1`, the request is admitted and `T` is decremented by 1.
3. Otherwise the request is refused with HTTP `429 Too Many
   Requests` and a `Retry-After` header equal to `ceil(1 / R)` seconds.

## 3. Configuration

Per-tenant capacity and refill rates are loaded from a configuration
file at startup and reloaded on `SIGHUP`. A tenant with no explicit
configuration receives the default policy of `C = 100` and `R = 10`.

## 4. Persistence

Tideline is a stateless front-end component. Token-bucket state lives
in process memory and is **not** persisted across restarts. On
restart, every bucket is reinitialized to its configured capacity.

## 5. Observability

Tideline must export the following metrics in Prometheus format:

- `tideline_requests_total{tenant, outcome}` — counter, labelled by
  outcome (`admitted` or `refused`).
- `tideline_tokens_current{tenant}` — gauge, current token count per
  tenant.
- `tideline_refills_total{tenant}` — counter, total refill events.

## 6. Security

Tideline must not echo tenant identifiers in its error responses
beyond the `Retry-After` header. The full tenant identifier may
appear only in structured logs that the operator controls.

## 7. Performance

Per-request overhead must remain below 200 microseconds at the 99th
percentile under a synthetic load of 10k requests per second across
1k distinct tenants.
