# Lighthouse — Edge Telemetry Gateway Specification

*Version 1.2. Engineering reference.*

Lighthouse is an edge-resident telemetry gateway. It accepts metric
and trace payloads from on-premise agents, batches them, and forwards
them to one or more downstream sinks (S3, OTLP collectors, Kafka). It
is designed to keep operating across network partitions and to bound
its memory footprint under back-pressure.

## 1. Goals

1. Bounded memory under sustained downstream outages (≤ 256 MB by default).
2. At-least-once forwarding semantics; the gateway never silently drops
   a payload it has acknowledged to an agent.
3. Single-binary deployment with no external coordination service.

Lighthouse is **not** a metrics database. Long-term storage and query
belong downstream.

## 2. Architecture

### 2.1 Components

Lighthouse is composed of three components running inside a single
process:

- **AgentEndpoint** — accepts HTTP and gRPC payloads from agents.
  Each request is authenticated against a static token list.
- **SpoolStore** — a bounded on-disk ring buffer that durably holds
  accepted payloads until the SinkBus has flushed them downstream.
- **SinkBus** — the fan-out engine that forwards spooled payloads to
  every configured sink with per-sink retry and back-off.

### 2.2 Flow

A typical request follows this path:

1. An agent POSTs a payload to AgentEndpoint.
2. AgentEndpoint validates the auth token and the payload schema.
3. The payload is appended to SpoolStore.
4. AgentEndpoint returns `202 Accepted` to the agent.
5. SinkBus reads from SpoolStore in FIFO order and dispatches to each
   sink. On success the payload's spool entry is freed.

## 3. Back-pressure

When SpoolStore reaches 90% of its capacity, AgentEndpoint enters
**throttled mode**: it returns `429 Too Many Requests` with a
`Retry-After` header. Agents are expected to honor this and back off
exponentially.

If SpoolStore reaches 100% capacity, AgentEndpoint returns
`503 Service Unavailable`. Lighthouse never silently drops an
accepted payload; clients must retry on `503`.

## 4. Security

- AgentEndpoint requires TLS 1.3 in production. HTTP and gRPC both
  pin server certificates.
- Static auth tokens are rotated weekly by an out-of-band process.
- Lighthouse must not log payload bodies. Headers and metadata only.

## 5. Operations

Lighthouse exposes a `/metrics` endpoint in Prometheus format. The
following gauges are required:

- `lighthouse_spool_bytes` — current on-disk spool size.
- `lighthouse_sink_lag_seconds` — per-sink oldest-pending-payload age.
- `lighthouse_throttled` — 1 if AgentEndpoint is currently throttling.

Operators are expected to alert on `sink_lag_seconds > 300` for any
sink.
