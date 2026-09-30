# Qwen Flash Next HLA cache persistence on CUDA

The HLA connector detects `compressed_key_cache` (also
`compress_key_cache`) and `raw_key_cache` entries in the vLLM cache specs.
It associates them with the main attention layer by layer index, not by the
position of a tensor in `kv_cache_tensors`.

## Fixed-size records

For the 24-tensor layout, the twelve main shared tensors and twelve auxiliary
tensors become twelve logical rows. Each row contributes three segments:

```
[main page, P_i bytes | compressed keys, C_i bytes | raw ring, R_i bytes]
```

Every stored record contains all rows and has exactly
`sum(P_i + C_i + R_i)` bytes. Segment sizes need not be equal. Copy lengths
come from the registered block views; block strides come from the tensor
strides. Compressed and raw addresses use their own group's block tables,
even when their views share an underlying allocation.

- Historical attention records contain main KV, compressed keys, and a zero
  raw segment.
- A complete attention boundary record contains the same KV and compressed
  keys, plus the request's current raw ring (including any packed position
  fields).
- Mamba state records contain the main state pages and zero auxiliary segments.

Real GPU zero buffers are used for padding. They and the discard buffer are
registered with the store alongside the cache buffers.

## Last raw ring only

No intermediate ring snapshots are generated. A complete record is written
only if the actual scheduled end is a full attention block boundary and does
not exceed the request's persistence range. For example, a step ending at
1024 can persist the raw ring at 1024; a step ending at 1000 cannot attach its
ring to the historical block ending at 768.

The final attention record uses its original key and carries the valid raw
ring directly. No boundary key or duplicate attention record is generated.
Earlier records retain zero padding in their raw segments.

This deliberately leaves raw validity unresolved for deduplicating stores:
if a key already contains zero padding, a later write of valid raw may be
skipped. Key presence alone cannot establish raw validity. Such a hit is not
a guarantee of correct recovery; overwrite/validity handling is deferred.

QSA uses the original base seed and group hash rules without adding a format
version, cache specs, or physical page schema. With identical hash inputs,
attention and Mamba keys match the original HLA keys; keys do not distinguish
the old and new storage layouts.

## Prefix-cache recovery

Lookup first checks the historical attention chain. A usable recovery point
must also have the original attention record and every Mamba state at the
same LCM-aligned position, using the original HLA lookup algorithm without
a QSA-specific lookup branch.

Load uses the original attention keys. Only the final record restores the raw
segment; there is currently no independent raw-validity check. Padding in all other records is
written to discard memory, never to the live raw or compressed caches.

## Scope and tradeoffs

- CUDA GQA/MHA, one main full-attention group, `pipeline_parallel_size=1`.
- Compressed token blocks must cover the same token range as main attention
  blocks; the raw cache must use an aligned `CircularBufferSpec`.
- Main rows must each pair one full-attention layer with its auxiliary caches.
- QSA uses whole-block transfers, including when the configured HLA connector
  class is layerwise. Dump occurs after forward completion. Other HLA layouts
  retain their existing paths.
- Loads currently submit and wait for one record at a time. This bounds
  scratch memory and avoids concurrent writes to the discard buffer, at the
  cost of load parallelism. Dump still batches records.
- No additional boundary keys are written. Intermediate raw snapshots and
  decode-state persistence are not added.

CPU contract tests (no PyTorch/vLLM installation required):

```
python -m unittest discover -s test/suites/Unit -p test_hla_qsa.py -v
```

These tests exercise production layout/planning and worker transfer methods
with NumPy-backed buffers and byte gather/scatter. GPU validation is still
required with the deployed vLLM version: cold-store recovery, chunked prefill,
TP, MRoPE where applicable, event synchronization, and output comparison with
prefix caching disabled.
