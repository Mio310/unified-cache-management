# Cache Store DRAM tail eviction: initial design

Status: proposed. The policy is not yet integrated with the Cache Store.
The connector can construct and forward relations under an internal, disabled
flag. The native pipeline deliberately rejects explicit relations until the
shared metadata and real slot-claim protocol are connected. Adding relations to
the public C++ TaskDesc changes its ABI and requires rebuilding all participating
Store plugins and the pipeline together. No production enablement is available.

## Scope

The initial implementation targets vLLM GLM-5.2 with MLA and layerwise transfers,
and the ordinary non-MLA direct path. CP, PP and Mamba-like state caches are
outside its scope. The existing Clock policy remains the default. Unsupported
configurations must not silently use a partially enforced tail policy.

## Eviction semantics

Block identities form a parent-child graph supplied by the connector; the store
cannot derive parent identities from their hashes. A block is resident while it
owns at least one allocated shard, including a loading or preallocated shard.
Only direct parent-child relationships between resident blocks protect the
parent. A missing intermediate block does not protect an ancestor; when that
block becomes resident, its direct resident relations must be re-established.

Only leaf blocks are eligible for eviction. An accessed bit gives a leaf a Clock
second chance. An eviction claims all currently resident shards of a block
before removing any of them; if a claim fails it releases the claims and retries
elsewhere. Pinned shards and shards still used by device or backend I/O must not
be recycled.

Slots belong to rank partitions. When no eligible leaf frees the target rank,
bounded cascading eviction may remove leaves from a target block's resident
subtree until the target itself becomes a leaf. Every successful leaf eviction
is committed independently. Scanning, total evictions and elapsed time are
bounded; pinning, concurrent insertion or budget exhaustion can still prevent
progress. No fallback may silently evict a non-leaf.

## Metadata and lifecycle

The metadata budget defaults to 256 MiB per Cache shared domain and includes
block nodes, relations, shard indexes and candidate indexes. Slot metadata is
accounted separately. Nodes include identity, parent identity, resident-child
count, resident shard locations, generation, state and heat. A reverse parent
index connects children already resident when a parent arrives. Generation
checks protect against stale references when metadata or slots are reused.

The connector supplies an explicit parent or root for each distinct task block.
Submission registers relationships transactionally and retains their lifetime
through asynchronous work. A conflicting relationship or exhausted metadata
budget is reported rather than silently dropping a relationship. Garbage
collection must not reclaim metadata needed by a resident block or in-flight
operation. It reclaims unreferenced entries and stale candidate records within
the budget.

## Layerwise and concurrency requirements

Each layer is submitted as a separate task; duplicate registrations must be
idempotent and must not increment child counts multiple times. MLA distributed
loads choose a preferred filler rank, and MLA dump keys are partitioned among
ranks. Relations must be generated from the original request chain before any
rank slicing, and attached to the exact block keys submitted by that rank.
Preallocation observes the same relation and residency protocol. In the first
integration it should be disabled for this policy until its zero-reference
Loading slots can be safely distinguished from active I/O: a preallocated slot
may never acquire an owner and must not become permanently unevictable.

Insertion, child-count updates and the transition to an evicting state require a
single synchronization protocol. New references must validate state and
generation after pinning. Locks must have a documented order; no tree or bucket
lock is held during I/O or while waiting for a slot. Cancelling a task cannot
release memory until outstanding device copies and backend operations no longer
use it. Allocation failure propagates through the existing submit / Check / Wait
contract; the existing timeout-and-drain implementation alone does not guarantee
bounded completion of external I/O.

The existing prefix lookup tests shard zero rather than readiness of every
layer. A shard-zero result must not be interpreted as a complete DRAM block;
partial residency and backend misses need explicit verification before enabling
this policy. Existing dump completion can precede backend completion, whose
failure is logged later. A submitted backend dump is not proof of persistence;
the implementation must define and test the behavior of failed or absent
backend copies before relying on them for recovery.

## Observability and validation

Log the configured budget, actual and peak metadata bytes, resident/partial
blocks, candidate lengths, GC counts and bytes, registration failures, scans,
cascade lengths and cross-rank rollbacks. Aggregate or rate-limit runtime logs.

Tests must cover parent-late insertion, gaps, duplicate and conflicting
relations, part-resident blocks, per-layer and multi-rank concurrency, pinned
shards, preallocation, metadata exhaustion, cascade budgets, cancellation and
backend/device lifetime. Benchmark throughput, hit rate, allocation P95/P99,
metadata footprint and cascade cost against the existing Clock implementation.
