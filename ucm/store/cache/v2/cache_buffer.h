/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "ctrl_layout.h"
#include "ctrl_strategy.h"
#include "data_strategy.h"
#include "global_config.h"
#include "metrics_api.h"
#include "status/status.h"
#include "type/types.h"

namespace UC::Cache2 {

class Buffer {
    friend struct BufferTestAccess;

    using SlotMeta = CtrlLayout::SlotMeta;
    using State = SlotMeta::State;

    CtrlStrategy ctrl_;
    DataStrategy data_;
    size_t myRank_{kInvalid};
    size_t rankCount_{0};
    size_t slotsPerRank_{0};
    size_t slotSize_{0};
    size_t shardsPerBlock_{0};
    size_t bucketCount_{0};
    size_t reservedSlots_{0};
    bool metricsRegistered_{false};
    std::string metricsDomainId_;
    std::unique_ptr<TailMetadata> tail_;
    size_t relationWaitTimeoutMs_{30000};

    static constexpr size_t kPinSpinFast{64};

public:
    class RelationToken {
        friend class Buffer;
        Buffer* buffer_{nullptr};
        std::vector<TailMetadata::Ref> refs_;
        explicit RelationToken(Buffer* buffer) : buffer_(buffer) {}
        void Reset() noexcept
        {
            if (buffer_ && buffer_->tail_) {
                for (auto ref : refs_) { (void)buffer_->tail_->Release(ref); }
                auto* header = buffer_->ctrl_.Layout().Hdr();
                if (header->tailMaxInflightBatches != 0) {
                    header->tailInflightBatches.fetch_sub(1, std::memory_order_release);
                }
            }
            refs_.clear();
            buffer_ = nullptr;
        }
    public:
        RelationToken() = default;
        RelationToken(const RelationToken&) = delete;
        RelationToken& operator=(const RelationToken&) = delete;
        RelationToken(RelationToken&& other) noexcept
            : buffer_(other.buffer_), refs_(std::move(other.refs_)) { other.buffer_ = nullptr; }
        RelationToken& operator=(RelationToken&& other) noexcept
        {
            if (this != &other) {
                Reset();
                buffer_ = other.buffer_;
                refs_ = std::move(other.refs_);
                other.buffer_ = nullptr;
            }
            return *this;
        }
        ~RelationToken() { Reset(); }
    };

    Expected<RelationToken> RegisterRelations(const Detail::TaskDesc& desc)
    {
        if (!tail_) {
            if (!desc.blockRelations.empty()) { return Status::Unsupported(); }
            return RelationToken{};
        }
        std::unordered_set<Detail::BlockId, Detail::BlockIdHasher> related;
        related.reserve(desc.blockRelations.size());
        for (const auto& rel : desc.blockRelations) { related.insert(rel.child); }
        for (const auto& shard : desc) {
            if (related.find(shard.owner) == related.end()) {
                return Status::InvalidParam("missing cache2 block relation");
            }
        }
        if (related.empty()) { return RelationToken{}; }
        // A single batch cannot be registered atomically if its distinct IDs
        // exceed the entire node table, even after evicting every resident.
        const auto nodeCapacity = tail_->NodeCapacity();
        if (related.size() > nodeCapacity) {
            return Status{
                Status::NoSpace().Underlying(),
                fmt::format("cache2 tail relation batch exceeds node capacity: "
                            "requested_unique_blocks={} node_capacity={}",
                            related.size(), nodeCapacity)};
        }
        const auto now = std::chrono::steady_clock::now();
        const auto maxWait = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::time_point::max() - now).count();
        const auto timeout = std::min(relationWaitTimeoutMs_, static_cast<size_t>(maxWait));
        const auto deadline = now + std::chrono::milliseconds(timeout);
        auto* header = ctrl_.Layout().Hdr();
        if (header->tailMaxBatchBlocks != 0 && related.size() > header->tailMaxBatchBlocks) {
            return Status{
                Status::NoSpace().Underlying(),
                fmt::format("cache2 tail relation batch exceeds configured limit: "
                            "requested_unique_blocks={} max_batch_blocks={}",
                            related.size(), header->tailMaxBatchBlocks)};
        }
        const auto batchLimit = header->tailMaxInflightBatches;
        if (batchLimit != 0) {
            auto& inflight = header->tailInflightBatches;
            for (;;) {
                auto active = inflight.load(std::memory_order_relaxed);
                if (active < batchLimit &&
                    inflight.compare_exchange_weak(active, active + 1, std::memory_order_acquire,
                                                   std::memory_order_relaxed)) { break; }
                if (std::chrono::steady_clock::now() >= deadline) {
                    return Status{
                        Status::Timeout().Underlying(),
                        fmt::format("cache2 tail relation admission deadline exceeded: "
                                    "requested_unique_blocks={} max_inflight_batches={}",
                                    related.size(), batchLimit)};
                }
                if (active >= batchLimit) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
        }
        RelationToken token(this);
        token.refs_.reserve(desc.blockRelations.size());
        size_t reclaimedNodes = 0;
        for (const auto& rel : desc.blockRelations) {
            TailMetadata::Ref ref;
            for (;;) {
                const auto result = tail_->Register(rel.child, rel.parent, rel.isRoot, ref);
                if (result == TailMetadata::Result::Ok) { break; }
                if (result == TailMetadata::Result::Poisoned) {
                    return Status::Error("cache2 tail metadata is poisoned");
                }
                if (result != TailMetadata::Result::Full && result != TailMetadata::Result::Busy) {
                    return Status::InvalidParam();
                }
                const auto current = std::chrono::steady_clock::now();
                if (current >= deadline) {
                    const bool full = result == TailMetadata::Result::Full;
                    const auto status = full ? Status::NoSpace() : Status::Timeout();
                    const auto elapsedMs =
                        std::chrono::duration_cast<std::chrono::milliseconds>(current - now).count();
                    return Status{
                        status.Underlying(),
                        fmt::format("cache2 tail relation registration deadline exceeded: "
                                    "last_result={} requested_unique_blocks={} node_capacity={} "
                                    "registered_relations={} elapsed_ms={} reclaimed_nodes={}",
                                    full ? "full" : "busy", related.size(), nodeCapacity,
                                    token.refs_.size(), elapsedMs, reclaimedNodes)};
                }
                // A task lease prevents collecting its node; never evict such
                // a block merely to make room for this registration batch.
                const bool progressed = result == TailMetadata::Result::Full && EvictTail(true);
                reclaimedNodes += static_cast<size_t>(progressed);
                if (!progressed) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            }
            token.refs_.push_back(ref);
        }
        return std::move(token);
    }

    struct EvictionSlot {
        size_t index;
        size_t offset;
    };

    // Caller must first mark the block evicting in shared tail metadata and
    // pass its complete set of resident slots. Registering a new shard must
    // honor that marker. commitMetadata must atomically commit that exact
    // eviction and leave metadata unchanged on failure; it must not acquire
    // bucket locks. This primitive is deliberately not called by CLOCK.
    // No bucket lock or claim may be held by the caller on entry. Slots are
    // claimed by index; unique bucket stripes are tried in address order.
    // Failed claims/locks/validation
    // or a failed metadata commit leave all bindings intact.
    Status TryEvictSlots(const Detail::BlockId& blockId, std::vector<EvictionSlot> slots,
                         const std::function<Status()>& commitMetadata)
    {
        if (slots.empty() || !commitMetadata || bucketCount_ == 0) {
            return Status::InvalidParam("invalid cache2 eviction request");
        }
        auto& layout = ctrl_.Layout();
        std::sort(slots.begin(), slots.end(), [](const auto& a, const auto& b) {
            return a.index < b.index;
        });
        std::vector<CtrlLayout::BucketLock*> locks;
        locks.reserve(slots.size());
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].index >= layout.SlotCount() ||
                (i != 0 && slots[i - 1].index == slots[i].index)) {
                return Status::InvalidParam("invalid or duplicate cache2 eviction slot");
            }
            locks.push_back(layout.LockOf(HashKey(blockId, slots[i].offset)));
        }
        std::sort(locks.begin(), locks.end(), std::less<CtrlLayout::BucketLock*>{});
        locks.erase(std::unique(locks.begin(), locks.end()), locks.end());

        size_t claimed = 0;
        size_t locked = 0;
        auto rollback = [&] {
            while (locked != 0) { locks[--locked]->Unlock(); }
            while (claimed != 0) {
                layout.SlotMetaArr()[slots[--claimed].index].reference.store(
                    0, std::memory_order_release);
            }
        };
        auto releaseGuard = [&](CtrlLayout*) { rollback(); };
        std::unique_ptr<CtrlLayout, decltype(releaseGuard)> guard(&layout, releaseGuard);
        size_t readySlots = 0;
        for (const auto& slot : slots) {
            auto& meta = layout.SlotMetaArr()[slot.index];
            size_t expected = 0;
            if (!meta.reference.compare_exchange_strong(expected, kSlotClaimed,
                                                         std::memory_order_acq_rel)) {
                rollback();
                return Status::Retry();
            }
            ++claimed;
        }
        for (auto* lock : locks) {
            if (!lock->TryLock()) { rollback(); return Status::Retry(); }
            ++locked;
        }
        for (const auto& slot : slots) {
            const auto bucket = HashKey(blockId, slot.offset);
            const auto& meta = layout.SlotMetaArr()[slot.index];
            if (!MatchKey(meta, bucket, blockId, slot.offset) ||
                Lookup(layout, bucket, blockId, slot.offset) != slot.index) {
                rollback();
                return Status::Retry();
            }
            if (meta.state.load(std::memory_order_acquire) == State::Ready) { ++readySlots; }
        }
        Status result = Status::Error("cache2 eviction metadata commit threw");
        for (const auto& slot : slots) {
            Remove(layout, HashKey(blockId, slot.offset), slot.index);
        }
        try {
            result = commitMetadata();
        } catch (...) {
            // The commit contract requires that an exception leaves metadata unchanged.
        }
        if (result.Failure()) {
            for (const auto& slot : slots) {
                MoveTo(layout, HashKey(blockId, slot.offset), slot.index);
            }
        } else {
            layout.Hdr()->usedSlots.fetch_sub(slots.size(), std::memory_order_relaxed);
        }
        rollback();
        if (result.Success() && readySlots != 0) {
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dram_evicted_bytes_total"),
                                 static_cast<double>(readySlots) * slotSize_);
        }
        return result;
    }

    class Handle {
        friend class Buffer;
        Buffer* buf_;
        size_t slotIdx_;
        bool owner_;

        Handle(Buffer* buf, size_t slotIdx, bool owner)
            : buf_(buf), slotIdx_(slotIdx), owner_(owner)
        {
        }

    public:
        Handle() = delete;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& o) noexcept : buf_(o.buf_), slotIdx_(o.slotIdx_), owner_(o.owner_)
        {
            o.buf_ = nullptr;
            o.slotIdx_ = kInvalid;
            o.owner_ = false;
        }
        Handle& operator=(Handle&& o) noexcept
        {
            Handle tmp(std::move(o));
            Swap(tmp);
            return *this;
        }
        ~Handle()
        {
            // A moved-from handle no longer owns a slot reference.
            if (buf_ != nullptr) {
                if (owner_ && GetState() == State::Loading) { buf_->MarkFailed(slotIdx_); }
                buf_->Release(slotIdx_);
            }
        }
        // Accessors require a handle that has not been moved from.
        bool Owner() const { return owner_; }
        bool HostAccessible() const { return buf_->data_.HostAccessibleOf(slotIdx_); }
        size_t SlotIndex() const { return slotIdx_; }
        size_t Segment() const { return slotIdx_ / buf_->slotsPerRank_; }
        void* Data() { return buf_->data_.DataAt(slotIdx_); }
        void* DeviceData() { return buf_->data_.DeviceDataAt(slotIdx_); }
        // The device-visible alias of this slot's host memory. Only transfer
        // modes that build device descriptors (SDMA Direct) use it; never for
        // CPU access.
        void* HostMappedData() { return buf_->data_.HostMappedDeviceDataAt(slotIdx_); }
        CtrlLayout::SlotMeta::State GetState() const { return buf_->GetState(slotIdx_); }
        void MarkReady()
        {
            if (Owner()) { buf_->MarkReady(slotIdx_); }
        }
        void MarkFailed()
        {
            if (Owner()) { buf_->MarkFailed(slotIdx_); }
        }
        // A failed device synchronization does not prove DMA has stopped.
        // Transfer this handle's existing pin into a permanent quarantine pin;
        // repeated calls and destruction after quarantine are no-ops.
        void Quarantine() noexcept
        {
            if (!buf_) { return; }
            auto& meta = buf_->ctrl_.Layout().SlotMetaArr()[slotIdx_];
            if (owner_ && meta.state.load(std::memory_order_acquire) != State::Ready) {
                meta.state.store(State::Failed, std::memory_order_release);
            }
            meta.quarantined.store(1, std::memory_order_release);
            buf_ = nullptr;
            slotIdx_ = kInvalid;
            owner_ = false;
        }

    private:
        void Swap(Handle& o) noexcept
        {
            std::swap(buf_, o.buf_);
            std::swap(slotIdx_, o.slotIdx_);
            std::swap(owner_, o.owner_);
        }
    };

    Buffer() = default;
    ~Buffer()
    {
        if (metricsRegistered_) { Reporter().Remove(this); }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Status Setup(const Config& cfg)
    {
        relationWaitTimeoutMs_ = cfg.timeoutMs == 0 ? 30000 : cfg.timeoutMs;
        if (cfg.localRankSize == 0 || cfg.localRankSize > kMaxRanks) {
            return Status::InvalidParam("invalid cache2 local rank size({})", cfg.localRankSize);
        }
        if (cfg.loadExclusiveBufferNumber % cfg.localRankSize != 0) {
            return Status::InvalidParam(
                "cache2 load exclusive buffer number({}) must be divisible by ranks({})",
                cfg.loadExclusiveBufferNumber, cfg.localRankSize);
        }
        auto s = ctrl_.Setup(cfg);
        if (s.Failure()) { return s; }
        auto& layout = ctrl_.Layout();
        if (cfg.enableDramTailEviction) {
            if (layout.TailMemory() == nullptr) { return Status::Error("tail metadata missing"); }
            tail_ = std::make_unique<TailMetadata>(layout.TailMemory());
        }
        rankCount_ = layout.RankCount();
        slotsPerRank_ = layout.SlotsPerRank();
        slotSize_ = layout.SlotSize();
        shardsPerBlock_ = cfg.blockSize / slotSize_;
        bucketCount_ = layout.BucketCount();
        reservedSlots_ = cfg.loadExclusiveBufferNumber / rankCount_;

        if (cfg.deviceId < 0) { return Status::OK(); }
        if (reservedSlots_ >= slotsPerRank_) {
            return Status::InvalidParam(
                "cache2 reserved slots per rank({}) must be less than slots per rank({})",
                reservedSlots_, slotsPerRank_);
        }
        /* Cache2 currently identifies the worker-local partition by deviceId. */
        myRank_ = static_cast<size_t>(cfg.deviceId);
        myRank_ = myRank_ % rankCount_;
        layout.InitSlotRange(myRank_);
        DataOptions dataOptions;
        dataOptions.domainId = cfg.uniqueId;
        dataOptions.deviceId = cfg.deviceId;
        dataOptions.myRank = myRank_;
        dataOptions.slotSize = slotSize_;
        dataOptions.slotsPerRank = slotsPerRank_;
        dataOptions.setupTimeoutMs = cfg.dataSetupTimeoutMs;
        dataOptions.deviceNumaNode = cfg.detectedNumaNode;
        /* Derived from the transfer mode: only a path that puts the host
         * segment into a device descriptor needs its device-visible alias. */
        dataOptions.requireHostDeviceAddress = cfg.sdmaDirect;
        if (cfg.fallbackNumaRank.has_value()) {
            dataOptions.fallbackNumaRank = static_cast<size_t>(*cfg.fallbackNumaRank);
        }
        if (!cfg.detectedNumaNode.has_value() && !cfg.fallbackNumaRank.has_value()) {
            // Older standalone callers know only their local Cache domain.
            // vLLM supplies machine-wide hints above, including local DP workers.
            dataOptions.fallbackNumaRank = myRank_;
            UC_WARN("cache2 NUMA hints absent; using cache-domain rank={} ranks={}", myRank_,
                    rankCount_);
        }
        if (cfg.fallbackNumaRankCount.has_value()) {
            dataOptions.fallbackNumaRankCount = static_cast<size_t>(*cfg.fallbackNumaRankCount);
        }
        s = data_.Setup(layout, dataOptions);
        if (s.Failure()) { return s; }
        if (myRank_ == 0) {
            try {
                metricsDomainId_ = cfg.uniqueId;
                Reporter().Add(this);
                metricsRegistered_ = true;
            } catch (const std::exception& error) {
                return Status::Error(
                    std::string("failed to register cache2 capacity metrics: ") + error.what());
            }
        }
        return Status::OK();
    }

    // Requires successful worker-side Setup; observers must not call Get.
    // Waits for a slot reference and always returns a valid handle, not necessarily Ready.
    Handle Get(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false,
               size_t preferredRank = kInvalid)
    {
        assert(myRank_ < rankCount_);
        assert(preferredRank == kInvalid || preferredRank < rankCount_);
        assert(reservedSlots_ < slotsPerRank_);
        auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        auto attempts = usable > std::numeric_limits<size_t>::max() / 2
                            ? std::numeric_limits<size_t>::max()
                            : 2 * usable;
        for (;;) {
            bool owner = false;
            auto slot =
                TryAcquireSlot(blockId, offset, allowReserved, attempts, owner, preferredRank);
            if (slot != kInvalid) { return Handle{this, slot, owner}; }
            std::this_thread::yield();
        }
    }

    // Bounded/cancellable variant of Get. A cancellation observed before an
    // acquisition attempt returns Retry; expiration returns Timeout. If either
    // races with a successful attempt, the handle may still be returned and
    // the caller remains responsible for handling cancellation and I/O safely.
    // Attempts are intentionally small so cancellation can be observed between
    // scans, but scheduling and an individual attempt do not have a hard bound.
    Expected<Handle> GetUntil(const Detail::BlockId& blockId, size_t offset,
                              std::chrono::steady_clock::time_point deadline,
                              const std::atomic_bool& cancelled, bool allowReserved = false,
                              size_t preferredRank = kInvalid)
    {
        assert(myRank_ < rankCount_);
        assert(preferredRank == kInvalid || preferredRank < rankCount_);
        assert(reservedSlots_ < slotsPerRank_);
        const auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        constexpr size_t kMaxAttemptsPerScan = 64;
        const auto attempts = usable > kMaxAttemptsPerScan / 2 ? kMaxAttemptsPerScan : 2 * usable;
        bool metadataFull = false;
        for (;;) {
            if (cancelled.load(std::memory_order_acquire)) { return Status::Retry(); }
            if (std::chrono::steady_clock::now() >= deadline) {
                return metadataFull ? Status::NoSpace() : Status::Timeout();
            }
            bool owner = false;
            Status error = Status::OK();
            auto slot = TryAcquireSlot(blockId, offset, allowReserved, attempts, owner, preferredRank,
                                       &error);
            if (slot != kInvalid) { return Handle{this, slot, owner}; }
            if (tail_ && error == Status::NoSpace()) {
                metadataFull = true;
                if (!EvictTail(true)) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
                continue;
            }
            if (error.Failure()) { return error; }
            std::this_thread::yield();
        }
    }

    void Prealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false,
                  size_t preferredRank = kInvalid)
    {
        if (myRank_ == kInvalid) { return; }
        assert(preferredRank == kInvalid || preferredRank < rankCount_);
        if (preferredRank != kInvalid && myRank_ != preferredRank) { return; }
        auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        if (usable == 0) { return; }
        auto attempts = usable > std::numeric_limits<size_t>::max() / 2
                            ? std::numeric_limits<size_t>::max()
                            : 2 * usable;
        TryPrealloc(blockId, offset, allowReserved, attempts);
    }

    bool Exist(const Detail::BlockId& blockId, size_t offset)
    {
        if (bucketCount_ == 0) { return false; }
        auto iBucket = HashKey(blockId, offset);
        auto& layout = ctrl_.Layout();
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            bool owner = false;
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner, false)) {
                if (!TouchTail(blockId, iNode, owner, nullptr)) { return false; }
                Release(iNode);
                return true;
            }
        }
        auto* lock = layout.LockOf(iBucket);
        lock->Lock();
        iNode = Lookup(layout, iBucket, blockId, offset);
        auto found = iNode != kInvalid &&
                     !layout.SlotMetaArr()[iNode].quarantined.load(std::memory_order_acquire);
        if (found && tail_) {
            TailMetadata::Ref ref;
            auto result = tail_->Acquire(blockId, ref);
            if (result == TailMetadata::Result::Ok) {
                result = tail_->Touch(ref);
                const auto released = tail_->Release(ref);
                if (released != TailMetadata::Result::Ok) { result = released; }
            }
            found = result == TailMetadata::Result::Ok;
        }
        if (found) { layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed); }
        lock->Unlock();
        return found;
    }

    void Touch(const Detail::BlockId* blocks, size_t num)
    {
        if (blocks == nullptr || bucketCount_ == 0) { return; }
        auto& layout = ctrl_.Layout();
        for (size_t i = 0; i < num; ++i) {
            bool found = false;
            // Shards of one block now live in separate hash buckets.
            for (size_t offset = 0; offset < shardsPerBlock_; ++offset) {
                auto iBucket = HashKey(blocks[i], offset);
                auto iNode = LookupOptimistic(layout, iBucket, blocks[i], offset);
                if (iNode != kInvalid) {
                    layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed);
                    found = true;
                }
            }
            if (found && tail_) {
                TailMetadata::Ref ref;
                if (tail_->Acquire(blocks[i], ref) == TailMetadata::Result::Ok) {
                    (void)tail_->Touch(ref);
                    (void)tail_->Release(ref);
                }
            }
        }
    }

    size_t MyRank() const { return myRank_; }
    size_t RankCount() const { return rankCount_; }
    size_t SlotsPerRank() const { return slotsPerRank_; }
    size_t BucketCount() const { return bucketCount_; }

private:
    // The caller already pinned this slot. A concurrent selected eviction will
    // either fail to claim it or make Touch return Busy; never return an
    // untracked hit as a valid tail-mode handle.
    bool TouchTail(const Detail::BlockId& id, size_t slot, bool owner, Status* error)
    {
        if (!tail_) { return true; }
        TailMetadata::Ref ref;
        auto result = tail_->Acquire(id, ref);
        if (result == TailMetadata::Result::Ok) {
            result = tail_->Touch(ref);
            const auto released = tail_->Release(ref);
            if (released != TailMetadata::Result::Ok) { result = released; }
        }
        if (result == TailMetadata::Result::Ok) { return true; }
        if (owner && GetState(slot) == State::Loading) { MarkFailed(slot); }
        Release(slot);
        if (error && result != TailMetadata::Result::Busy) {
            *error = result == TailMetadata::Result::Full ? Status::NoSpace()
                     : result == TailMetadata::Result::Poisoned
                         ? Status::Error("cache2 tail metadata is poisoned")
                         : Status::InvalidParam();
        }
        return false;
    }

    // Attempt a direct leaf first; if its rank differs, the bounded cascade
    // picks a leaf whose resident ancestor can eventually free this rank.
    bool EvictTail(bool reclaimMetadata = false)
    {
        if (!tail_ || myRank_ >= rankCount_) { return false; }
        TailMetadata::Ref selection;
        const auto budget = std::min<uint32_t>(tail_->NodeCapacity(), 256);
        auto result = tail_->Select(static_cast<uint32_t>(myRank_), budget, false, selection,
                                    reclaimMetadata, reclaimMetadata);
        if (!reclaimMetadata && result == TailMetadata::Result::NotFound) {
            result = tail_->Select(static_cast<uint32_t>(myRank_), budget, true, selection);
        }
        if (result != TailMetadata::Result::Ok) { return false; }
        bool committed = false;
        auto abort = [&](TailMetadata*) {
            if (!committed) { (void)tail_->AbortEvict(selection); }
        };
        std::unique_ptr<TailMetadata, decltype(abort)> guard(tail_.get(), abort);
        try {
            Detail::BlockId id{};
            if (tail_->EvictionId(selection, id) != TailMetadata::Result::Ok) { return false; }
            std::vector<TailMetadata::Shard> snapshot;
            if (tail_->GetShards(selection, snapshot) != TailMetadata::Result::Ok) { return false; }
            std::vector<EvictionSlot> slots;
            slots.reserve(snapshot.size());
            for (const auto& shard : snapshot) {
                if (shard.slot >= ctrl_.Layout().SlotCount() ||
                    shard.rank != shard.slot / slotsPerRank_) { return false; }
                const auto offset = ctrl_.Layout().SlotMetaArr()[shard.slot].key[2].load(
                    std::memory_order_acquire);
                slots.push_back({static_cast<size_t>(shard.slot), offset});
            }
            if (slots.empty()) { return false; }
            const auto status = TryEvictSlots(id, std::move(slots), [&] {
                const auto commit = tail_->CommitEvict(selection);
                if (commit != TailMetadata::Result::Ok) { return Status::Retry(); }
                committed = true;
                return Status::OK();
            });
            return status.Success();
        } catch (...) {
            // The selection guard clears the evicting flag on allocation errors.
            return false;
        }
    }

    // Metrics aggregates process-local Gauge writes without domain labels. One
    // reporter thread must therefore sum all rank-zero domains in this process.
    class CapacityReporter {
        std::mutex mutex_;
        std::condition_variable wake_;
        std::unordered_map<std::string, std::unordered_set<Buffer*>> domains_;
        std::unordered_map<std::string, uint64_t> lastTailGc_;
        std::chrono::steady_clock::time_point lastTailLog_{std::chrono::steady_clock::now()};
        bool stopping_{false};
        bool dirty_{false};
        std::thread thread_;

    public:
        CapacityReporter() : thread_([this] { Run(); }) {}
        ~CapacityReporter()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            wake_.notify_one();
            if (thread_.joinable()) { thread_.join(); }
        }
        void Add(Buffer* buffer)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto existing = domains_.find(buffer->metricsDomainId_);
            if (existing == domains_.end()) {
                // Prepare the set before publishing it: an allocation failure
                // must never leave an empty domain for SnapshotLocked().
                std::unordered_set<Buffer*> members;
                members.insert(buffer);
                domains_.emplace(buffer->metricsDomainId_, std::move(members));
            } else {
                existing->second.insert(buffer);
            }
            dirty_ = true;
            wake_.notify_one();
        }
        void Remove(Buffer* buffer)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = domains_.find(buffer->metricsDomainId_);
            if (it != domains_.end()) {
                if (it->second.size() == 1 && it->second.count(buffer) && buffer->tail_) {
                    const auto stats = buffer->tail_->GetStats();
                    if (stats.status == TailMetadata::Result::Ok) {
                        ObserveTailGcLocked(it->first, stats.gc);
                    }
                    lastTailGc_.erase(it->first);
                }
                it->second.erase(buffer);
                if (it->second.empty()) { domains_.erase(it); }
            }
            dirty_ = true;
            wake_.notify_one();
        }
        std::pair<double, double> Snapshot()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return SnapshotLocked();
        }

    private:
        void ObserveTailGcLocked(const std::string& domain, uint64_t count)
        {
            auto& previous = lastTailGc_[domain];
            // A new shared region may reuse an id after the previous one exits.
            if (count < previous) { previous = 0; }
            if (count > previous) {
                Metrics::UpdateStats("cache_dram_tail_metadata_gc_total",
                                     static_cast<double>(count - previous));
                previous = count;
            }
        }

        void ReportTailStatsLocked(bool log)
        {
            double capacity = 0;
            double used = 0;
            bool complete = true;
            for (const auto& [domain, buffers] : domains_) {
                auto* buffer = *buffers.begin();
                if (!buffer->tail_) { continue; }
                const auto stats = buffer->tail_->GetStats();
                if (stats.status != TailMetadata::Result::Ok) {
                    complete = false;
                    UC_WARN("cache2 tail metadata stats unavailable for domain={}", domain);
                    continue;
                }
                capacity += static_cast<double>(stats.capacityBytes);
                used += static_cast<double>(stats.usedBytes);
                ObserveTailGcLocked(domain, stats.gc);
                if (log) {
                    UC_INFO_UNLIMITED("cache_dram_tail_metadata domain={} used_bytes={} "
                                      "capacity_bytes={} nodes={} shards={} gc_total={} full_total={}",
                                      domain, stats.usedBytes, stats.capacityBytes, stats.nodes,
                                      stats.shards, stats.gc, stats.failed);
                }
            }
            if (complete) {
                Metrics::UpdateStats("cache_dram_tail_metadata_capacity_bytes", capacity);
                Metrics::UpdateStats("cache_dram_tail_metadata_used_bytes", used);
            }
        }

        std::pair<double, double> SnapshotLocked()
        {
            double capacity = 0;
            double used = 0;
            // The registry lock keeps the mapped control memory alive here.
            for (const auto& [domain, buffers] : domains_) {
                (void)domain;
                auto* buffer = *buffers.begin();
                auto& layout = buffer->ctrl_.Layout();
                capacity += static_cast<double>(layout.SlotCount()) * buffer->slotSize_;
                used += static_cast<double>(layout.Hdr()->usedSlots.load(
                            std::memory_order_relaxed)) * buffer->slotSize_;
            }
            return {capacity, used};
        }

        void Run()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!stopping_) {
                wake_.wait_for(lock, std::chrono::seconds(10),
                               [this] { return stopping_ || dirty_; });
                if (stopping_) { break; }
                dirty_ = false;
                const auto [capacity, used] = SnapshotLocked();
                // Avoid reporter-thread function-local statics: their destruction
                // could precede the reporter's static destructor and join.
                Metrics::UpdateStats("cache_dram_capacity_bytes", capacity);
                Metrics::UpdateStats("cache_dram_used_bytes", used);
                const auto now = std::chrono::steady_clock::now();
                const bool log = now - lastTailLog_ >= std::chrono::seconds(60);
                if (log) { lastTailLog_ = now; }
                ReportTailStatsLocked(log);
            }
        }
    };

    static CapacityReporter& Reporter()
    {
        // Ensure the metrics singleton outlives the reporter thread.
        (void)Metrics::Metrics::GetInstance();
        static CapacityReporter reporter;
        return reporter;
    }

    void ReportCapacityStats()
    {
        auto& layout = ctrl_.Layout();
        const auto usedSlots = layout.Hdr()->usedSlots.load(std::memory_order_relaxed);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dram_capacity_bytes"),
                             static_cast<double>(layout.SlotCount()) * slotSize_);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dram_used_bytes"),
                             static_cast<double>(usedSlots) * slotSize_);
    }

    static void BlockKeyWords(const Detail::BlockId& blockId, size_t (&words)[2])
    {
        static_assert(sizeof(Detail::BlockId) == sizeof(words));
        std::memcpy(words, blockId.data(), sizeof(blockId));
    }

    size_t HashKey(const Detail::BlockId& blockId, size_t offset) const
    {
        static Detail::BlockIdHasher blockIdHasher;
        static std::hash<size_t> shardHasher;
        // Keep the cc core's block/shard mix, with Cache2's power-of-two buckets.
        constexpr auto goldenSection = 0x9e3779b97f4a7c15ULL;
        size_t h1 = blockIdHasher(blockId);
        size_t h2 = shardHasher(offset);
        return (h1 ^ (h2 + goldenSection + (h1 << 6) + (h1 >> 2))) & (bucketCount_ - 1);
    }

    static void StoreKey(SlotMeta& meta, const Detail::BlockId& blockId, size_t offset)
    {
        size_t words[2];
        BlockKeyWords(blockId, words);
        meta.key[0].store(words[0], std::memory_order_relaxed);
        meta.key[1].store(words[1], std::memory_order_relaxed);
        meta.key[2].store(offset, std::memory_order_relaxed);
    }

    static bool MatchBlock(const SlotMeta& meta, size_t iBucket, const Detail::BlockId& blockId)
    {
        if (meta.hash.load(std::memory_order_acquire) != iBucket) { return false; }
        size_t words[2];
        BlockKeyWords(blockId, words);
        return meta.key[0].load(std::memory_order_acquire) == words[0] &&
               meta.key[1].load(std::memory_order_acquire) == words[1];
    }

    static bool MatchKey(const SlotMeta& meta, size_t iBucket, const Detail::BlockId& blockId,
                         size_t offset)
    {
        return MatchBlock(meta, iBucket, blockId) &&
               meta.key[2].load(std::memory_order_acquire) == offset;
    }

    // On success, the caller owns one reference and must transfer it into a Handle.
    // kInvalid means retry; owner is only meaningful on success.
    size_t TryAcquireSlot(const Detail::BlockId& blockId, size_t offset, bool allowReserved,
                          size_t attempts, bool& owner, size_t preferredRank = kInvalid,
                          Status* error = nullptr)
    {
        auto iBucket = HashKey(blockId, offset);
        auto& layout = ctrl_.Layout();
        const bool mayOwn = preferredRank == kInvalid || myRank_ == preferredRank;
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner, mayOwn)) {
                if (!TouchTail(blockId, iNode, owner, error)) { return kInvalid; }
                return iNode;
            }
        }
        /* All ranks may pin a ready or already-owned shared key. Distributed
         * loads use the same preferred rank for Get and next-shard Prealloc,
         * so the selected filler owns its preallocated slot. */
        if (!mayOwn) { return kInvalid; }

        auto* targetLock = layout.LockOf(iBucket);
        if (!targetLock->TryLock()) { return kInvalid; }
        iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            auto pinned = PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner);
            targetLock->Unlock();
            return pinned && TouchTail(blockId, iNode, owner, error) ? iNode : kInvalid;
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved, attempts, 1, error);
        targetLock->Unlock();
        if (iNode != kInvalid) { owner = true; }
        if (iNode == kInvalid && tail_ && (!error || error->Success())) { EvictTail(); }
        return iNode;
    }

    bool TryPrealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved,
                     size_t attempts)
    {
        auto iBucket = HashKey(blockId, offset);
        auto& layout = ctrl_.Layout();
        auto* targetLock = layout.LockOf(iBucket);
        if (!targetLock->TryLock()) { return false; }
        auto iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed);
            targetLock->Unlock();
            return true;
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved, attempts, 0);
        targetLock->Unlock();
        return iNode != kInvalid;
    }

    bool PinHit(CtrlLayout& layout, size_t iNode, size_t iBucket, const Detail::BlockId& blockId,
                size_t offset, size_t spinBudget, bool& owner, bool takeOwnership = true)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        for (size_t spin = 0; spin < spinBudget;) {
            if (!MatchKey(*meta, iBucket, blockId, offset)) { return false; }
            if (meta->quarantined.load(std::memory_order_acquire)) { return false; }
            auto reference = meta->reference.load(std::memory_order_acquire);
            if (reference == kSlotClaimed) {
                ++spin;
                std::this_thread::yield();
                continue;
            }
            if (reference == kSlotClaimed - 1) { return false; }
            auto state = meta->state.load(std::memory_order_acquire);
            /* A non-owner may observe Ready or Failed. Loading with no
             * reference is a preallocated slot that still needs its selected
             * rank to claim and fill it. */
            if (!takeOwnership && reference == 0 && state == State::Loading) { return false; }
            if (!meta->reference.compare_exchange_weak(reference, reference + 1,
                                                       std::memory_order_acq_rel)) {
                ++spin;
                continue;
            }
            if (!MatchKey(*meta, iBucket, blockId, offset) ||
                meta->quarantined.load(std::memory_order_acquire)) {
                meta->reference.fetch_sub(1, std::memory_order_release);
                return false;
            }
            state = meta->state.load(std::memory_order_acquire);
            owner = takeOwnership && reference == 0 && state != State::Ready;
            if (owner && state == State::Failed) {
                meta->state.store(State::Loading, std::memory_order_release);
            }
            meta->accessed.store(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    template <std::memory_order Order>
    size_t LookupT(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId,
                   size_t offset)
    {
        auto iNode = layout.Buckets()[iBucket].load(Order);
        size_t walked = 0;
        while (iNode != kInvalid && iNode < layout.SlotCount() && walked++ < layout.SlotCount()) {
            auto* meta = &layout.SlotMetaArr()[iNode];
            if (MatchKey(*meta, iBucket, blockId, offset)) { return iNode; }
            iNode = meta->next.load(Order);
        }
        return kInvalid;
    }

    size_t Lookup(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId, size_t offset)
    {
        return LookupT<std::memory_order_relaxed>(layout, iBucket, blockId, offset);
    }

    size_t LookupOptimistic(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId,
                            size_t offset)
    {
        return LookupT<std::memory_order_acquire>(layout, iBucket, blockId, offset);
    }

    size_t Alloc(CtrlLayout& layout, const Detail::BlockId& blockId, size_t offset, size_t iBucket,
                 bool allowReserved, size_t attempts, size_t initialReference, Status* error = nullptr)
    {
        for (size_t scan = 0; scan < attempts; ++scan) {
            auto iNode = FetchNode(layout, allowReserved);
            if (iNode == kInvalid) { continue; }
            auto* meta = &layout.SlotMetaArr()[iNode];
            size_t expected = 0;
            if (!meta->reference.compare_exchange_strong(expected, kSlotClaimed,
                                                         std::memory_order_acq_rel)) {
                meta->accessed.store(1, std::memory_order_relaxed);
                continue;
            }

            auto oldBucket = meta->hash.load(std::memory_order_acquire);
            if (tail_) {
                // Tail mode may only bind an actually unlinked slot. It must
                // never use CLOCK to replace an existing, tracked shard.
                if (oldBucket != kInvalid) {
                    meta->reference.store(0, std::memory_order_release);
                    continue;
                }
                TailMetadata::Ref ref;
                auto registered = tail_->Acquire(blockId, ref);
                if (registered == TailMetadata::Result::Ok) {
                    registered = tail_->AddShard(ref, iNode, static_cast<uint32_t>(myRank_));
                    const auto released = tail_->Release(ref);
                    if (released != TailMetadata::Result::Ok) {
                        // Metadata may already refer to this slot. Keep it
                        // permanently unavailable when the domain is poisoned.
                        meta->quarantined.store(1, std::memory_order_release);
                        meta->reference.store(1, std::memory_order_release);
                        if (error) { *error = Status::Error("cache2 tail metadata is poisoned"); }
                        return kInvalid;
                    }
                }
                if (registered != TailMetadata::Result::Ok) {
                    meta->reference.store(0, std::memory_order_release);
                    if (error && registered != TailMetadata::Result::Busy) {
                        *error = registered == TailMetadata::Result::Full ? Status::NoSpace()
                                 : registered == TailMetadata::Result::Poisoned
                                     ? Status::Error("cache2 tail metadata is poisoned")
                                     : Status::InvalidParam();
                    }
                    return kInvalid;
                }
            }
            const bool evictsReady = oldBucket != kInvalid &&
                meta->state.load(std::memory_order_acquire) == State::Ready;
            if (oldBucket != iBucket) {
                if (oldBucket != kInvalid) {
                    auto* oldLock = layout.LockOf(oldBucket);
                    auto sameStripe = oldLock == layout.LockOf(iBucket);
                    if (!sameStripe && !oldLock->TryLock()) {
                        meta->accessed.store(1, std::memory_order_relaxed);
                        meta->reference.store(0, std::memory_order_release);
                        continue;
                    }
                    Remove(layout, oldBucket, iNode);
                    if (!sameStripe) { oldLock->Unlock(); }
                }
                StoreKey(*meta, blockId, offset);
                meta->state.store(State::Loading, std::memory_order_relaxed);
                MoveTo(layout, iBucket, iNode);
            } else {
                StoreKey(*meta, blockId, offset);
                meta->state.store(State::Loading, std::memory_order_relaxed);
            }
            if (oldBucket == kInvalid) {
                layout.Hdr()->usedSlots.fetch_add(1, std::memory_order_relaxed);
            } else if (evictsReady) {
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dram_evicted_bytes_total"),
                                     static_cast<double>(slotSize_));
            }
            meta->accessed.store(1, std::memory_order_relaxed);
            meta->reference.store(initialReference, std::memory_order_release);
            return iNode;
        }
        return kInvalid;
    }

    size_t FetchNode(CtrlLayout& layout, bool allowReserved)
    {
        auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        if (usable == 0 || myRank_ == kInvalid) { return kInvalid; }
        auto iNode = layout.NextClockSlot(myRank_, usable);
        if (tail_ && layout.SlotMetaArr()[iNode].hash.load(std::memory_order_acquire) != kInvalid) {
            return kInvalid;
        }
        if (layout.SlotMetaArr()[iNode].accessed.exchange(0, std::memory_order_relaxed) != 0) {
            return kInvalid;
        }
        return iNode;
    }

    static void MoveTo(CtrlLayout& layout, size_t iBucket, size_t iNode)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        auto& head = layout.Buckets()[iBucket];
        auto next = head.load(std::memory_order_relaxed);
        meta->next.store(next, std::memory_order_relaxed);
        meta->prev.store(kInvalid, std::memory_order_relaxed);
        if (next != kInvalid) {
            layout.SlotMetaArr()[next].prev.store(iNode, std::memory_order_relaxed);
        }
        meta->hash.store(iBucket, std::memory_order_release);
        head.store(iNode, std::memory_order_release);
    }

    static void Remove(CtrlLayout& layout, size_t iBucket, size_t iNode)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        auto prev = meta->prev.load(std::memory_order_relaxed);
        auto next = meta->next.load(std::memory_order_relaxed);
        if (prev != kInvalid) {
            layout.SlotMetaArr()[prev].next.store(next, std::memory_order_release);
        }
        if (next != kInvalid) {
            layout.SlotMetaArr()[next].prev.store(prev, std::memory_order_relaxed);
        }
        if (layout.Buckets()[iBucket].load(std::memory_order_relaxed) == iNode) {
            layout.Buckets()[iBucket].store(next, std::memory_order_release);
        }
        meta->prev.store(kInvalid, std::memory_order_relaxed);
        meta->next.store(kInvalid, std::memory_order_relaxed);
        meta->hash.store(kInvalid, std::memory_order_release);
    }

    void Release(size_t slotIdx)
    {
        ctrl_.Layout().SlotMetaArr()[slotIdx].reference.fetch_sub(1, std::memory_order_release);
    }

    State GetState(size_t slotIdx) const
    {
        auto& meta = ctrl_.Layout().SlotMetaArr()[slotIdx];
        if (meta.quarantined.load(std::memory_order_acquire)) { return State::Failed; }
        return meta.state.load(std::memory_order_acquire);
    }

    void MarkReady(size_t slotIdx)
    {
        ctrl_.Layout().SlotMetaArr()[slotIdx].state.store(State::Ready, std::memory_order_release);
    }

    void MarkFailed(size_t slotIdx)
    {
        ctrl_.Layout().SlotMetaArr()[slotIdx].state.store(State::Failed, std::memory_order_release);
    }
};

}  // namespace UC::Cache2
