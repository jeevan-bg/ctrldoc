# Aurora — Distributed Cache System Specification

*Version 0.4. Internal engineering reference.*

This document describes Aurora, a distributed in-memory cache for read-heavy workloads. It is designed to run across 4–256 nodes on commodity hardware. Aurora is not a database; it stores transient state with a configurable retention policy.

## 1. Overview

Aurora provides a key-value cache with optional secondary indexes. Clients connect over a binary protocol and address keys directly. Internally, Aurora partitions the key space across nodes using a consistent-hashing scheme and replicates each partition to a configurable number of peers.

The system is designed around three goals:

1. Predictable read latency under load.
2. Bounded operational complexity (single binary, no external coordination service).
3. Graceful degradation when nodes fail.

Aurora is **not** suitable for workloads requiring strong durability or transactional guarantees across keys. For those, use a purpose-built database.

## 2. Architecture

### 2.1 Components

Aurora is composed of four components, each running as part of every node:

- **NodeManager (NM)** — the top-level coordinator on a node. Owns lifecycle, configuration, and inter-component dispatch. Every other component reports to its local NodeManager.
- **GossipBus** — a SWIM-style gossip protocol that maintains the cluster membership view. Each NodeManager subscribes to its local GossipBus for membership change events.
- **ShardRing** — the consistent-hash ring that maps keys to nodes. ShardRing refines the classical consistent-hashing approach by adding virtual nodes (typically 256 per physical node) to reduce skew.
- **CacheStore** — the per-node storage layer. Backed by an in-memory hash map with a configurable eviction policy (LRU by default).

A client never talks to CacheStore directly. All traffic enters through NodeManager, which consults ShardRing to route the request.

### 2.2 Data flow

A read request follows this path:

1. Client opens a session and presents a `ClientToken`.
2. NodeManager validates the token and consults ShardRing.
3. If the local node owns the partition, the request is served from CacheStore.
4. If a peer owns the partition, the request is forwarded; the local node optionally caches the response.
5. The response is returned to the client with a freshness timestamp.

Writes follow the same path but are quorum-acknowledged across N replicas (default N=3).

## 3. Consistency Model

### 3.1 Eventually consistent reads

By default, reads are **eventually consistent**. A read may return a value that is stale by up to the gossip convergence interval (typically 200ms across a 32-node cluster). Clients that require fresher data can request a "read-after-write" upgrade, which routes the read to the partition leader.

### 3.2 Strongly consistent writes

Writes are linearizable within a single partition. A write returns only after a quorum of replicas has acknowledged. Cross-partition writes are not atomic; users requiring multi-key atomicity should compose operations at the application layer.

## 4. Fault Tolerance

Aurora tolerates the failure of any minority of replicas without data loss. When a node fails:

1. GossipBus detects the failure within 3 gossip rounds (≈600ms).
2. NodeManager removes the node from ShardRing.
3. Partitions previously held by the failed node are re-replicated from surviving peers.

A simultaneous failure of a majority of replicas for a partition is **unrecoverable**; the partition becomes unavailable until enough peers return.

Operators may configure replication factor N up to 7. Higher N increases availability at the cost of write latency.

## 5. Security

All client traffic typically authenticates via a `ClientToken`. The token may be issued by an external identity provider and should usually be rotated periodically. Aurora itself does not issue tokens; it validates them.

Inter-node traffic uses mutual TLS. Certificate provisioning is out of scope for this specification.

The token format and signature algorithm are determined by the issuer.

## 6. Performance

Aurora targets the following on a 32-node cluster with 16-core nodes and 64GB RAM each:

- 1M reads/sec aggregate.
- 200k writes/sec aggregate.
- P99 read latency under 5ms.
- Linearizable reads available to clients that opt in.

Memory overhead per cached entry is approximately 64 bytes beyond the key + value payload.

## 7. Operations

### 7.1 Scaling

To add a node, deploy the binary with the cluster's join address. The new node will register with GossipBus, and ShardRing will rebalance partitions onto it over the next few minutes.

To remove a node, mark it as draining; partitions migrate off it within a bounded window before the binary shuts down.

### 7.2 Upgrades

Rolling upgrades are supported within a minor version. Cross-major upgrades require a fresh cluster.

## 8. Glossary

- **CacheStore** — per-node in-memory key-value store.
- **ClientToken** — opaque credential presented by clients for authentication.
- **GossipBus** — SWIM-style membership protocol.
- **NodeManager** — per-node coordinator.
- **Partition** — a range of the hash space owned by one or more replica nodes.
- **ShardRing** — consistent-hash ring with virtual nodes.
