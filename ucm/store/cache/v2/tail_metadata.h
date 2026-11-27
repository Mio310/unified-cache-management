/**
 * MIT License
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace UC::Cache2 {

// A bounded, relocatable metadata prototype. The caller owns and shares the
// memory region; View itself is process-local and must never be stored there.
// This class is not connected to Buffer's eviction path.
class TailMetadata {
public:
    using Id = std::array<std::byte, 16>;
    static constexpr uint32_t kNil = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kMaxRanks = 128;
    // ticket is nonzero only on a selection. It distinguishes successive
    // eviction attempts on the same live node from stale work after Abort.
    struct Ref { uint32_t index{kNil}; uint64_t generation{0}, ticket{0}; };
    enum class Result { Ok, Full, Conflict, Invalid, Busy, NotFound };
    struct Shard { uint64_t slot; uint32_t rank; };
    struct Stats { uint32_t nodes; uint32_t shards; uint64_t gc; uint64_t failed; };

private:
    struct Header {
        std::atomic<uint32_t> lock{0};
        uint32_t nodes{0}, shards{0}, buckets{0};
        uint32_t freeNode{kNil}, freeShard{kNil}, cursor{0};
        uint32_t usedNodes{0}, usedShards{0};
        uint64_t nextGeneration{1}, nextTicket{1}, gc{0}, failed{0};
    };
    struct Node {
        Id id{}, parent{};
        uint64_t generation{0}, ticket{0};
        uint32_t hashNext{kNil}, parentNext{kNil}, shardHead{kNil};
        uint32_t refs{0}, children{0}, shards{0};
        bool used{false}, root{false}, evicting{false}, accessed{false};
    };
    struct ShardNode { uint64_t slot{0}; uint32_t rank{0}, next{kNil}; };
    Header* h_;
    Node* nodes_;
    ShardNode* shards_;
    uint32_t* ids_;
    uint32_t* parents_;

    static size_t Align(size_t n, size_t a) { return (n + a - 1) / a * a; }
    static size_t Hash(const Id& id)
    {
        size_t value = static_cast<size_t>(1469598103934665603ULL);
        for (auto b : id) { value = (value ^ std::to_integer<uint8_t>(b)) * 1099511628211ULL; }
        return value;
    }
    class Guard {
        Header* h_;
    public:
        explicit Guard(Header* h) : h_(h)
        {
            uint32_t expected;
            do { expected = 0; } while (!h_->lock.compare_exchange_weak(
                expected, 1, std::memory_order_acquire, std::memory_order_relaxed));
        }
        ~Guard() { h_->lock.store(0, std::memory_order_release); }
    };
    uint32_t Find(const Id& id) const
    {
        for (auto i = ids_[Hash(id) % h_->buckets]; i != kNil; i = nodes_[i].hashNext) {
            if (nodes_[i].id == id) { return i; }
        }
        return kNil;
    }
    bool Valid(Ref r) const
    {
        return r.index < h_->nodes && nodes_[r.index].used &&
               nodes_[r.index].generation == r.generation;
    }
    Ref MakeRef(uint32_t i) const { return {i, nodes_[i].generation}; }
    bool ValidEviction(Ref r) const
    {
        return Valid(r) && r.ticket != 0 && nodes_[r.index].evicting &&
               nodes_[r.index].ticket == r.ticket;
    }
    bool HasRank(uint32_t i, uint32_t rank) const
    {
        for (auto s = nodes_[i].shardHead; s != kNil; s = shards_[s].next) {
            if (shards_[s].rank == rank) { return true; }
        }
        return false;
    }
    void Unlink(uint32_t* head, uint32_t index, bool parent)
    {
        while (*head != kNil && *head != index) {
            head = parent ? &nodes_[*head].parentNext : &nodes_[*head].hashNext;
        }
        if (*head == index) { *head = parent ? nodes_[index].parentNext : nodes_[index].hashNext; }
    }
    void Collect(uint32_t i)
    {
        auto& n = nodes_[i];
        if (n.refs || n.shards || n.evicting) { return; }
        Unlink(&ids_[Hash(n.id) % h_->buckets], i, false);
        if (!n.root) { Unlink(&parents_[Hash(n.parent) % h_->buckets], i, true); }
        n.used = false;
        n.hashNext = h_->freeNode;
        h_->freeNode = i;
        --h_->usedNodes;
        ++h_->gc;
    }
    void BecameResident(uint32_t i)
    {
        auto& n = nodes_[i];
        if (!n.root) {
            auto p = Find(n.parent);
            if (p != kNil && nodes_[p].shards) { ++nodes_[p].children; }
        }
        for (auto c = parents_[Hash(n.id) % h_->buckets]; c != kNil; c = nodes_[c].parentNext) {
            if (nodes_[c].parent == n.id && nodes_[c].shards) { ++n.children; }
        }
    }
    void BecameAbsent(uint32_t i)
    {
        auto& n = nodes_[i];
        if (!n.root) {
            auto p = Find(n.parent);
            if (p != kNil && nodes_[p].shards) { --nodes_[p].children; }
        }
        n.children = 0;
    }

public:
    // Counts are explicit so all fixed allocations, including buckets, fit the budget.
    static size_t Bytes(uint32_t nodeCount, uint32_t shardCount, uint32_t buckets)
    {
        size_t size = Align(sizeof(Header), alignof(Node));
        size = Align(size + size_t(nodeCount) * sizeof(Node), alignof(ShardNode));
        size = Align(size + size_t(shardCount) * sizeof(ShardNode), alignof(uint32_t));
        return size + size_t(buckets) * 2 * sizeof(uint32_t);
    }
    static constexpr size_t Alignment() { return alignof(Header) > alignof(Node) ? alignof(Header) : alignof(Node); }

    // The creator initializes once before publishing. Each process binds its own view.
    static bool Initialize(void* memory, size_t bytes, uint32_t nodeCount,
                           uint32_t shardCount, uint32_t buckets)
    {
        if (!memory || !nodeCount || !shardCount || !buckets ||
            reinterpret_cast<uintptr_t>(memory) % Alignment() ||
            bytes < Bytes(nodeCount, shardCount, buckets)) { return false; }
        auto* h = new (memory) Header();
        h->nodes = nodeCount; h->shards = shardCount; h->buckets = buckets;
        TailMetadata view(memory);
        for (uint32_t i = 0; i < nodeCount; ++i) {
            new (&view.nodes_[i]) Node();
            view.nodes_[i].hashNext = i + 1 < nodeCount ? i + 1 : kNil;
        }
        for (uint32_t i = 0; i < shardCount; ++i) {
            new (&view.shards_[i]) ShardNode();
            view.shards_[i].next = i + 1 < shardCount ? i + 1 : kNil;
        }
        for (uint32_t i = 0; i < buckets; ++i) { view.ids_[i] = view.parents_[i] = kNil; }
        h->freeNode = h->freeShard = 0;
        return true;
    }
    explicit TailMetadata(void* memory) : h_(static_cast<Header*>(memory))
    {
        auto* base = static_cast<std::byte*>(memory);
        auto offset = Align(sizeof(Header), alignof(Node));
        nodes_ = reinterpret_cast<Node*>(base + offset);
        offset = Align(offset + size_t(h_->nodes) * sizeof(Node), alignof(ShardNode));
        shards_ = reinterpret_cast<ShardNode*>(base + offset);
        offset = Align(offset + size_t(h_->shards) * sizeof(ShardNode), alignof(uint32_t));
        ids_ = reinterpret_cast<uint32_t*>(base + offset);
        parents_ = ids_ + h_->buckets;
    }

    // Every successful registration owns a reference until Release. This is a
    // single-entry primitive: callers must roll back earlier entries on batch failure.
    Result Register(const Id& id, const Id& parent, bool root, Ref& out)
    {
        Guard guard(h_);
        auto found = Find(id);
        if (found != kNil) {
            auto& n = nodes_[found];
            if (n.root != root || (!root && n.parent != parent)) { return Result::Conflict; }
            if (n.refs == UINT32_MAX) { return Result::Full; }
            ++n.refs; out = MakeRef(found); return Result::Ok;
        }
        // Reject cycles even if the final link of a previously incomplete chain arrives later.
        if (!root) {
            auto current = parent;
            for (uint32_t depth = 0; depth <= h_->nodes; ++depth) {
                if (current == id) { return Result::Conflict; }
                auto p = Find(current);
                if (p == kNil || nodes_[p].root) { break; }
                current = nodes_[p].parent;
                if (depth == h_->nodes) { return Result::Conflict; }
            }
        }
        if (h_->freeNode == kNil || h_->nextGeneration == UINT64_MAX) {
            ++h_->failed; return Result::Full;
        }
        auto i = h_->freeNode;
        h_->freeNode = nodes_[i].hashNext;
        nodes_[i] = Node{};
        auto& n = nodes_[i];
        n.id = id; n.parent = parent; n.root = root; n.used = true; n.refs = 1;
        n.generation = h_->nextGeneration++;
        auto bucket = Hash(id) % h_->buckets;
        n.hashNext = ids_[bucket]; ids_[bucket] = i;
        if (!root) {
            bucket = Hash(parent) % h_->buckets;
            n.parentNext = parents_[bucket]; parents_[bucket] = i;
        }
        ++h_->usedNodes; out = MakeRef(i); return Result::Ok;
    }
    Result Release(Ref r)
    {
        Guard guard(h_);
        if (!Valid(r) || !nodes_[r.index].refs) { return Result::Invalid; }
        --nodes_[r.index].refs; Collect(r.index); return Result::Ok;
    }
    // The caller must reserve the slot before adding it and must never attach
    // the same slot to two blocks. A failed add must roll back that reservation.
    Result AddShard(Ref r, uint64_t slot, uint32_t rank)
    {
        Guard guard(h_);
        if (!Valid(r) || rank >= kMaxRanks) { return Result::Invalid; }
        auto& n = nodes_[r.index];
        if (n.evicting) { return Result::Busy; }
        for (auto s = n.shardHead; s != kNil; s = shards_[s].next) {
            if (shards_[s].slot == slot) { return shards_[s].rank == rank ? Result::Ok : Result::Conflict; }
        }
        if (!n.shards && !n.root) {
            auto p = Find(n.parent);
            if (p != kNil && nodes_[p].evicting) { return Result::Busy; }
        }
        if (h_->freeShard == kNil) { ++h_->failed; return Result::Full; }
        auto s = h_->freeShard; h_->freeShard = shards_[s].next;
        shards_[s] = {slot, rank, n.shardHead}; n.shardHead = s;
        if (n.shards++ == 0) { BecameResident(r.index); }
        n.accessed = true; ++h_->usedShards; return Result::Ok;
    }
    // Use for failed loads or preallocation cleanup after the actual slot is
    // safely unlinked. Last-shard removal may create an unprotected gap.
    Result RemoveShard(Ref r, uint64_t slot)
    {
        Guard guard(h_);
        if (!Valid(r)) { return Result::Invalid; }
        auto& n = nodes_[r.index];
        if (n.evicting) { return Result::Busy; }
        auto* link = &n.shardHead;
        while (*link != kNil && shards_[*link].slot != slot) { link = &shards_[*link].next; }
        if (*link == kNil) { return Result::NotFound; }
        auto s = *link;
        *link = shards_[s].next;
        shards_[s].next = h_->freeShard;
        h_->freeShard = s;
        --h_->usedShards;
        if (--n.shards == 0) { BecameAbsent(r.index); }
        Collect(r.index);
        return Result::Ok;
    }
    Result Touch(Ref r)
    {
        Guard guard(h_);
        if (!Valid(r)) { return Result::Invalid; }
        if (nodes_[r.index].evicting) { return Result::Busy; }
        nodes_[r.index].accessed = true; return Result::Ok;
    }
    // Scan at most budget node entries; each can additionally traverse a parent
    // chain and its shards. To prioritize direct reclamation, call with cascade
    // false first. This marks the result evicting: the caller must synchronize
    // with slot pins, then claim every real shard before unlinking any of them.
    // Cascade never jumps a missing (nonresident) direct parent.
    Result Select(uint32_t rank, uint32_t budget, bool cascade, Ref& out)
    {
        Guard guard(h_);
        if (rank >= kMaxRanks) { return Result::Invalid; }
        for (uint32_t scan = 0; scan < budget; ++scan) {
            auto i = h_->cursor++ % h_->nodes;
            auto& n = nodes_[i];
            if (!n.used || !n.shards || n.children || n.evicting) { continue; }
            bool suitable = HasRank(i, rank);
            auto j = i;
            for (uint32_t depth = 0; !suitable && cascade && depth < h_->nodes; ++depth) {
                if (nodes_[j].root) { break; }
                j = Find(nodes_[j].parent);
                if (j == kNil || !nodes_[j].shards) { break; }
                suitable = HasRank(j, rank);
            }
            if (!suitable) { continue; }
            if (n.accessed) { n.accessed = false; continue; }
            // Never wrap a ticket: otherwise an indefinitely delayed operation
            // could become valid again while the node is still resident.
            if (h_->nextTicket == UINT64_MAX) { ++h_->failed; return Result::Full; }
            n.evicting = true;
            n.ticket = h_->nextTicket++;
            out = {i, n.generation, n.ticket};
            return Result::Ok;
        }
        return Result::NotFound;
    }
    Result GetShard(Ref r, uint32_t ordinal, Shard& out)
    {
        Guard guard(h_);
        if (!ValidEviction(r)) { return Result::Invalid; }
        auto s = nodes_[r.index].shardHead;
        while (s != kNil && ordinal) { s = shards_[s].next; --ordinal; }
        if (s == kNil) { return Result::NotFound; }
        out = {shards_[s].slot, shards_[s].rank}; return Result::Ok;
    }
    Result AbortEvict(Ref r)
    {
        Guard guard(h_);
        if (!ValidEviction(r)) { return Result::Invalid; }
        nodes_[r.index].evicting = false;
        nodes_[r.index].ticket = 0;
        // The final task reference may have been released while evicting.
        Collect(r.index);
        return Result::Ok;
    }
    // Only call after all corresponding real slots have been claimed and removed.
    Result CommitEvict(Ref r)
    {
        Guard guard(h_);
        if (!ValidEviction(r) || nodes_[r.index].children) { return Result::Invalid; }
        auto& n = nodes_[r.index];
        BecameAbsent(r.index);
        while (n.shardHead != kNil) {
            auto s = n.shardHead; n.shardHead = shards_[s].next;
            shards_[s].next = h_->freeShard; h_->freeShard = s; --h_->usedShards;
        }
        n.shards = 0; n.evicting = false; n.ticket = 0; Collect(r.index); return Result::Ok;
    }
    Stats GetStats()
    {
        Guard guard(h_); return {h_->usedNodes, h_->usedShards, h_->gc, h_->failed};
    }
};
static_assert(std::atomic<uint32_t>::is_always_lock_free, "shared lock must be lock-free");
} // namespace UC::Cache2
