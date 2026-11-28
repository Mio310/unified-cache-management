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

    static constexpr size_t kPinSpinFast{64};

public:
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
        for (;;) {
            if (cancelled.load(std::memory_order_acquire)) { return Status::Retry(); }
            if (std::chrono::steady_clock::now() >= deadline) { return Status::Timeout(); }
            bool owner = false;
            auto slot = TryAcquireSlot(blockId, offset, allowReserved, attempts, owner, preferredRank);
            if (slot != kInvalid) { return Handle{this, slot, owner}; }
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
                Release(iNode);
                return true;
            }
        }
        auto* lock = layout.LockOf(iBucket);
        lock->Lock();
        iNode = Lookup(layout, iBucket, blockId, offset);
        auto found = iNode != kInvalid;
        if (found) { layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed); }
        lock->Unlock();
        return found;
    }

    void Touch(const Detail::BlockId* blocks, size_t num)
    {
        if (blocks == nullptr || bucketCount_ == 0) { return; }
        auto& layout = ctrl_.Layout();
        for (size_t i = 0; i < num; ++i) {
            // Shards of one block now live in separate hash buckets.
            for (size_t offset = 0; offset < shardsPerBlock_; ++offset) {
                auto iBucket = HashKey(blocks[i], offset);
                auto iNode = LookupOptimistic(layout, iBucket, blocks[i], offset);
                if (iNode != kInvalid) {
                    layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed);
                }
            }
        }
    }

    size_t MyRank() const { return myRank_; }
    size_t RankCount() const { return rankCount_; }
    size_t SlotsPerRank() const { return slotsPerRank_; }
    size_t BucketCount() const { return bucketCount_; }

private:
    // Metrics aggregates process-local Gauge writes without domain labels. One
    // reporter thread must therefore sum all rank-zero domains in this process.
    class CapacityReporter {
        std::mutex mutex_;
        std::condition_variable wake_;
        std::unordered_map<std::string, std::unordered_set<Buffer*>> domains_;
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
                          size_t attempts, bool& owner, size_t preferredRank = kInvalid)
    {
        auto iBucket = HashKey(blockId, offset);
        auto& layout = ctrl_.Layout();
        const bool mayOwn = preferredRank == kInvalid || myRank_ == preferredRank;
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner, mayOwn)) {
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
            return pinned ? iNode : kInvalid;
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved, attempts, 1);
        targetLock->Unlock();
        if (iNode != kInvalid) { owner = true; }
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
            if (!MatchKey(*meta, iBucket, blockId, offset)) {
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
                 bool allowReserved, size_t attempts, size_t initialReference)
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
        return ctrl_.Layout().SlotMetaArr()[slotIdx].state.load(std::memory_order_acquire);
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
