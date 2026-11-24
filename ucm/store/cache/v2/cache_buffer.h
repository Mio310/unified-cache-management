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

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <thread>
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
    std::atomic<size_t> occupiedSlots_{0};

    // Deeper prefix first, then the lower slot index. Victim choice is O(log n);
    // a full-rank scan under the stripe lock stalls once the partition is full.
    struct EvictKey {
        uint32_t position{0};
        size_t index{0};
    };
    struct EvictOrder {
        bool operator()(const EvictKey& lhs, const EvictKey& rhs) const
        {
            if (lhs.position != rhs.position) { return lhs.position > rhs.position; }
            return lhs.index < rhs.index;
        }
    };
    using EvictSet = std::set<EvictKey, EvictOrder>;
    std::mutex evictMu_;
    bool evictReady_{false};
    size_t evictRank_{kInvalid};
    size_t evictSlotsPerRank_{0};
    size_t evictReserved_{0};
    std::vector<size_t> evictFree_;
    std::vector<size_t> evictFreeReserved_;
    EvictSet evictHot_;
    EvictSet evictCold_;

    static constexpr size_t kPinSpinFast{64};

public:
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
        RebuildEvictIndex();
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
        auto s2 = data_.Setup(layout, dataOptions);
        if (s2.Success()) { PublishDramUsage(); }
        return s2;
    }

    // Requires successful worker-side Setup; observers must not call Get.
    // Waits for a slot reference and always returns a valid handle, not necessarily Ready.
    Handle Get(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false,
               size_t preferredRank = kInvalid,
               uint32_t prefixPosition = Detail::kUnmarkedPrefixDepth)
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
            auto slot = TryAcquireSlot(blockId, offset, allowReserved, attempts, owner,
                                       preferredRank, prefixPosition);
            if (slot != kInvalid) { return Handle{this, slot, owner}; }
            std::this_thread::yield();
        }
    }

    void Prealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false,
                  size_t preferredRank = kInvalid,
                  uint32_t prefixPosition = Detail::kUnmarkedPrefixDepth)
    {
        if (myRank_ == kInvalid) { return; }
        assert(preferredRank == kInvalid || preferredRank < rankCount_);
        if (preferredRank != kInvalid && myRank_ != preferredRank) { return; }
        auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        if (usable == 0) { return; }
        auto attempts = usable > std::numeric_limits<size_t>::max() / 2
                            ? std::numeric_limits<size_t>::max()
                            : 2 * usable;
        TryPrealloc(blockId, offset, allowReserved, attempts, prefixPosition);
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
        if (found) { MarkAccessedHot(layout.SlotMetaArr()[iNode], iNode); }
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
                if (iNode != kInvalid) { MarkAccessedHot(layout.SlotMetaArr()[iNode], iNode); }
            }
        }
    }

    size_t MyRank() const { return myRank_; }
    size_t RankCount() const { return rankCount_; }
    size_t SlotsPerRank() const { return slotsPerRank_; }
    size_t BucketCount() const { return bucketCount_; }

private:
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
                          uint32_t prefixPosition = Detail::kUnmarkedPrefixDepth)
    {
        auto iBucket = HashKey(blockId, offset);
        auto& layout = ctrl_.Layout();
        const bool mayOwn = preferredRank == kInvalid || myRank_ == preferredRank;
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner, mayOwn,
                       prefixPosition)) {
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
            auto pinned = PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner,
                                 true, prefixPosition);
            targetLock->Unlock();
            return pinned ? iNode : kInvalid;
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved, attempts, 1, prefixPosition);
        targetLock->Unlock();
        if (iNode != kInvalid) { owner = true; }
        return iNode;
    }

    bool TryPrealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved,
                     size_t attempts, uint32_t prefixPosition = Detail::kUnmarkedPrefixDepth)
    {
        auto iBucket = HashKey(blockId, offset);
        auto& layout = ctrl_.Layout();
        auto* targetLock = layout.LockOf(iBucket);
        if (!targetLock->TryLock()) { return false; }
        auto iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalid) {
            auto* meta = &layout.SlotMetaArr()[iNode];
            const auto oldPos = meta->position.load(std::memory_order_relaxed);
            RememberPrefix(*meta, prefixPosition, true);
            MarkAccessedHot(*meta, iNode, oldPos);
            targetLock->Unlock();
            return true;
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved, attempts, 0, prefixPosition);
        targetLock->Unlock();
        return iNode != kInvalid;
    }

    void NoteDramInstall(bool newlyOccupied, bool evicted)
    {
        if (newlyOccupied) { occupiedSlots_.fetch_add(1, std::memory_order_relaxed); }
        if (evicted) {
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dram_evict_blocks_total"), 1.0);
        }
        PublishDramUsage();
    }

    void PublishDramUsage() const
    {
        if (slotSize_ == 0 || slotsPerRank_ == 0 || myRank_ == kInvalid) { return; }
        const auto capacity = static_cast<double>(slotsPerRank_) * static_cast<double>(slotSize_);
        const auto used =
            static_cast<double>(occupiedSlots_.load(std::memory_order_relaxed)) *
            static_cast<double>(slotSize_);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dram_used_bytes"), used);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dram_capacity_bytes"), capacity);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dram_usage_ratio"),
                             capacity == 0.0 ? 0.0 : used / capacity);
    }

    static void RememberPrefix(SlotMeta& meta, uint32_t prefixPosition, bool refresh)
    {
        if (refresh && prefixPosition == Detail::kUnmarkedPrefixDepth) { return; }
        meta.position.store(prefixPosition, std::memory_order_relaxed);
    }

    bool PinHit(CtrlLayout& layout, size_t iNode, size_t iBucket, const Detail::BlockId& blockId,
                size_t offset, size_t spinBudget, bool& owner, bool takeOwnership = true,
                uint32_t prefixPosition = Detail::kUnmarkedPrefixDepth)
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
            const auto oldPos = meta->position.load(std::memory_order_relaxed);
            RememberPrefix(*meta, prefixPosition, true);
            MarkAccessedHot(*meta, iNode, oldPos);
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
                 bool allowReserved, size_t attempts, size_t initialReference,
                 uint32_t prefixPosition)
    {
        /* FetchNode scans the rank, so a handful of passes covers CAS races
         * and one second-chance pass. Get() retries if every slot is pinned. */
        const auto scans = std::min(attempts, size_t{8});
        for (size_t scan = 0; scan < scans; ++scan) {
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
            const bool sameKey =
                oldBucket == iBucket && MatchKey(*meta, iBucket, blockId, offset);
            if (oldBucket != iBucket) {
                if (oldBucket != kInvalid) {
                    auto* oldLock = layout.LockOf(oldBucket);
                    auto sameStripe = oldLock == layout.LockOf(iBucket);
                    if (!sameStripe && !oldLock->TryLock()) {
                        MarkAccessedHot(*meta, iNode);
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
            const auto oldPos = meta->position.load(std::memory_order_relaxed);
            RememberPrefix(*meta, prefixPosition, sameKey);
            meta->accessed.store(1, std::memory_order_relaxed);
            meta->reference.store(initialReference, std::memory_order_release);
            NoteDramInstall(oldBucket == kInvalid, oldBucket != kInvalid && !sameKey);
            NoteEvictInstalled(iNode, oldBucket == kInvalid, oldPos);
            return iNode;
        }
        return kInvalid;
    }

    size_t FetchNode(CtrlLayout& layout, bool allowReserved)
    {
        auto usable = slotsPerRank_ - (allowReserved ? 0 : reservedSlots_);
        if (usable == 0 || myRank_ == kInvalid) { return kInvalid; }
        std::lock_guard<std::mutex> lock(evictMu_);
        if (!EvictIndexCurrent()) { RebuildEvictIndexLocked(); }
        const auto begin = myRank_ * slotsPerRank_;
        const auto freeSlot = TakeFreeLocked(allowReserved);
        if (freeSlot != kInvalid) { return freeSlot; }
        return SelectOccupiedLocked(layout, begin, begin + usable);
    }

    bool EvictIndexCurrent() const
    {
        return evictReady_ && evictRank_ == myRank_ && evictSlotsPerRank_ == slotsPerRank_ &&
               evictReserved_ == reservedSlots_;
    }

    void RebuildEvictIndex()
    {
        std::lock_guard<std::mutex> lock(evictMu_);
        RebuildEvictIndexLocked();
    }

    void RebuildEvictIndexLocked()
    {
        evictHot_.clear();
        evictCold_.clear();
        evictFree_.clear();
        evictFreeReserved_.clear();
        evictReady_ = false;
        if (myRank_ == kInvalid || slotsPerRank_ == 0) { return; }
        auto& layout = ctrl_.Layout();
        const auto base = myRank_ * slotsPerRank_;
        const auto reserved = reservedSlots_ < slotsPerRank_ ? reservedSlots_ : slotsPerRank_;
        const auto usable = slotsPerRank_ - reserved;
        auto pushFree = [&](size_t limit, std::vector<size_t>& dest) {
            for (size_t n = limit; n > 0; --n) {
                const auto local = n - 1;
                const auto index = base + local;
                auto& meta = layout.SlotMetaArr()[index];
                if (meta.hash.load(std::memory_order_relaxed) != kInvalid) { continue; }
                dest.push_back(index);
            }
        };
        pushFree(usable, evictFree_);
        for (size_t n = slotsPerRank_; n > usable; --n) {
            const auto index = base + n - 1;
            auto& meta = layout.SlotMetaArr()[index];
            if (meta.hash.load(std::memory_order_relaxed) == kInvalid) {
                evictFreeReserved_.push_back(index);
            }
        }
        for (size_t local = 0; local < slotsPerRank_; ++local) {
            const auto index = base + local;
            auto& meta = layout.SlotMetaArr()[index];
            if (meta.hash.load(std::memory_order_relaxed) == kInvalid) { continue; }
            EvictKey key{meta.position.load(std::memory_order_relaxed), index};
            if (meta.accessed.load(std::memory_order_relaxed) != 0) {
                evictHot_.insert(key);
            } else {
                evictCold_.insert(key);
            }
        }
        evictRank_ = myRank_;
        evictSlotsPerRank_ = slotsPerRank_;
        evictReserved_ = reservedSlots_;
        evictReady_ = true;
    }

    size_t TakeFreeLocked(bool allowReserved)
    {
        if (!evictFree_.empty()) {
            const auto index = evictFree_.back();
            evictFree_.pop_back();
            return index;
        }
        if (allowReserved && !evictFreeReserved_.empty()) {
            const auto index = evictFreeReserved_.back();
            evictFreeReserved_.pop_back();
            return index;
        }
        return kInvalid;
    }

    EvictSet::iterator FindEvictSlot(EvictSet& from, EvictSet& other, CtrlLayout& layout,
                                     size_t begin, size_t end, bool wantHot)
    {
        for (auto it = from.begin(); it != from.end();) {
            if (it->index < begin || it->index >= end) {
                ++it;
                continue;
            }
            auto& meta = layout.SlotMetaArr()[it->index];
            if (meta.reference.load(std::memory_order_acquire) != 0) {
                ++it;
                continue;
            }
            if (meta.hash.load(std::memory_order_acquire) == kInvalid) {
                const auto index = it->index;
                it = from.erase(it);
                ReturnFreeLocked(index);
                continue;
            }
            const bool hot = meta.accessed.load(std::memory_order_relaxed) != 0;
            if (hot != wantHot) {
                const auto key = *it;
                it = from.erase(it);
                other.insert(key);
                continue;
            }
            return it;
        }
        return from.end();
    }

    size_t SelectOccupiedLocked(CtrlLayout& layout, size_t begin, size_t end)
    {
        auto hot = FindEvictSlot(evictHot_, evictCold_, layout, begin, end, true);
        auto cold = FindEvictSlot(evictCold_, evictHot_, layout, begin, end, false);
        /* A hotter accessed tail keeps its second chance. Shallower cold blocks
         * stay until that tail is cold too. Same depth may be taken. */
        if (hot != evictHot_.end() &&
            (cold == evictCold_.end() || hot->position > cold->position)) {
            const auto key = *hot;
            evictHot_.erase(hot);
            layout.SlotMetaArr()[key.index].accessed.store(0, std::memory_order_relaxed);
            evictCold_.insert(key);
            return kInvalid;
        }
        if (cold == evictCold_.end()) { return kInvalid; }
        return cold->index;
    }

    void ReturnFreeLocked(size_t index)
    {
        const auto base = myRank_ * slotsPerRank_;
        const auto reserved = reservedSlots_ < slotsPerRank_ ? reservedSlots_ : slotsPerRank_;
        const auto usable = slotsPerRank_ - reserved;
        if (index - base < usable) {
            evictFree_.push_back(index);
        } else {
            evictFreeReserved_.push_back(index);
        }
    }

    void MarkAccessedHot(SlotMeta& meta, size_t index)
    {
        const auto oldPos = meta.position.load(std::memory_order_relaxed);
        meta.accessed.store(1, std::memory_order_relaxed);
        NoteEvictHot(index, oldPos);
    }

    void MarkAccessedHot(SlotMeta& meta, size_t index, uint32_t oldPos)
    {
        meta.accessed.store(1, std::memory_order_relaxed);
        NoteEvictHot(index, oldPos);
    }

    void NoteEvictHot(size_t index, uint32_t oldPos)
    {
        std::lock_guard<std::mutex> lock(evictMu_);
        if (!evictReady_) { return; }
        evictHot_.erase(EvictKey{oldPos, index});
        evictCold_.erase(EvictKey{oldPos, index});
        const auto pos =
            ctrl_.Layout().SlotMetaArr()[index].position.load(std::memory_order_relaxed);
        evictHot_.insert(EvictKey{pos, index});
    }

    void NoteEvictInstalled(size_t index, bool wasEmpty, uint32_t oldPos)
    {
        std::lock_guard<std::mutex> lock(evictMu_);
        if (!evictReady_) { return; }
        if (!wasEmpty) {
            evictHot_.erase(EvictKey{oldPos, index});
            evictCold_.erase(EvictKey{oldPos, index});
        }
        const auto pos =
            ctrl_.Layout().SlotMetaArr()[index].position.load(std::memory_order_relaxed);
        evictHot_.insert(EvictKey{pos, index});
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
