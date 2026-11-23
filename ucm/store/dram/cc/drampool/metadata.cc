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
#include "metadata.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include "drampool_metrics.h"
#include "logger/logger.h"
#include "pos_eviction_policy.h"
#include "prefix_tail_eviction_policy.h"
#include "ttl_eviction_policy.h"

namespace UC::DramPool {
namespace {
std::unique_ptr<EvictionPolicy> CreateEvictionPolicy(EvictionPolicyType type)
{
    switch (type) {
        case EvictionPolicyType::TTL: return std::make_unique<TtlEvictionPolicy>();
        case EvictionPolicyType::POSITION: return std::make_unique<PosEvictionPolicy>();
        case EvictionPolicyType::PREFIX_TAIL: return std::make_unique<PrefixTailEvictionPolicy>();
        default: break;
    }
    UC_ERROR("CreateEvictionPolicy: invalid EvictionPolicyType {}, fallback to TTL.",
             static_cast<int>(type));
    return std::make_unique<TtlEvictionPolicy>();
}
}  // namespace

ShardMetadata::ShardMetadata(const MetadataConfig& config)
    : periodicEvictor_(CreateEvictionPolicy(config.periodicType)),
      deepEvictor_(CreateEvictionPolicy(config.deepType)),
      leaseTime_(config.leaseTime)
{
}

Status ShardMetadata::StoreBegin(const BlockId& key, EntryPtr entry)
{
    ReadWriteGuard lock(mtx_);
    if (metadata_.find(key) != metadata_.end()) {
        // A later dump can learn the real prefix depth. The unmarked sentinel
        // must not overwrite a depth that is already known.
        if (entry != nullptr && entry->position != UC::Detail::kUnmarkedPrefixDepth) {
            periodicEvictor_->UpdatePosition(key, entry->position);
            deepEvictor_->UpdatePosition(key, entry->position);
        }
        UC_INFO("ShardMetadata StoreBegin: key already exists.");
        return Status::DuplicateKey();
    }
    if (entry == nullptr || !entry->IsInitial()) {
        UC_ERROR("ShardMetadata StoreBegin: entry not in initial state.");
        return Status::InvalidParam();
    }
    auto st = periodicEvictor_->AddKey(key, entry);
    if (!st.Success()) {
        UC_ERROR("ShardMetadata StoreBegin: periodicEvictor AddKey failed.");
        return st;
    }
    st = deepEvictor_->AddKey(key, entry);
    if (!st.Success()) {
        UC_ERROR("ShardMetadata StoreBegin: deepEvictor AddKey failed, rollback periodicEvictor.");
        periodicEvictor_->DeleteKey(key);
        return st;
    }
    metadata_.emplace(key, std::move(entry));
    return Status::OK();
}

Status ShardMetadata::StoreEnd(const BlockId& key)
{
    ScopedTimer storeEndTimer(NAME_TO_METRIC_ID(kMetadataStoreendDurationMs));
    storeEndTimer.Arm();
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    auto& entry = it->second;
    return entry->TryMarkReady() ? Status::OK() : Status::Error();
}

Status ShardMetadata::LoadBegin(const BlockId& key, EntryPtr& entry)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    auto& existingEntry = it->second;
    if (!existingEntry->TryIncRef()) { return Status::Error(); }
    periodicEvictor_->AccessKey(key);
    deepEvictor_->AccessKey(key);
    entry = existingEntry;
    return Status::OK();
}

Status ShardMetadata::LoadEnd(const BlockId& key)
{
    ScopedTimer loadEndTimer(NAME_TO_METRIC_ID(kMetadataLoadendDurationMs));
    loadEndTimer.Arm();
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    auto& entry = it->second;
    return entry->TryDecRef() ? Status::OK() : Status::Error();
}

bool ShardMetadata::Exist(const BlockId& key)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return false; }
    auto& entry = it->second;
    return entry->TryMarkHit(std::chrono::system_clock::now() + leaseTime_);
}

bool ShardMetadata::Query(const BlockId& key) const
{
    ReadOnlyGuard lock(mtx_);
    return metadata_.find(key) != metadata_.end();
}

Status ShardMetadata::Delete(const BlockId& key)
{
    ReadWriteGuard lock(mtx_);
    if (metadata_.find(key) == metadata_.end()) { return Status::NotFound(); }
    periodicEvictor_->DeleteKey(key);
    deepEvictor_->DeleteKey(key);
    metadata_.erase(key);
    return Status::OK();
}

Status ShardMetadata::TryDelete(const BlockId& key, EntryPtr& entry)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    auto& existingEntry = it->second;
    if (!existingEntry->TryMarkDeleting()) { return Status::Error(); }
    entry = existingEntry;
    return Status::OK();
}

std::size_t ShardMetadata::GetKeyCnt() const noexcept
{
    ReadOnlyGuard lock(mtx_);
    return metadata_.size();
}

std::vector<EntryPtr> ShardMetadata::EvictPeriodic(double evict_ratio)
{
    ReadOnlyGuard lock(mtx_);
    return periodicEvictor_->GetEvictionResults(evict_ratio);
}

std::vector<EntryPtr> ShardMetadata::EvictDeep(double evict_ratio)
{
    ReadOnlyGuard lock(mtx_);
    return deepEvictor_->GetEvictionResults(evict_ratio);
}

DepthCandidate ShardMetadata::FindTail(std::size_t size, std::chrono::system_clock::time_point now)
{
    ReadOnlyGuard lock(mtx_);
    return deepEvictor_->FindDeepest(size, now);
}

std::size_t ShardMetadata::CountSize(std::size_t size) const
{
    ReadOnlyGuard lock(mtx_);
    return deepEvictor_->CountSize(size);
}

bool ShardMetadata::TryMarkForEvict(const EntryPtr& entry,
                                    std::chrono::system_clock::time_point now)
{
    ReadOnlyGuard lock(mtx_);
    if (entry == nullptr || metadata_.find(entry->key) == metadata_.end()) { return false; }
    return entry->TryMarkEvicting(now);
}

Status MetadataManager::Delete(const BlockId& key)
{
    EntryPtr entry;
    auto st = ShardOf(key).TryDelete(key, entry);
    if (!st.Success()) { return st; }
    auto freeSt = bufferManager_.Free(entry->size, entry->buffer.slot);
    if (!freeSt.Success()) {
        UC_ERROR("Delete: Free slot {} failed, status {}.", entry->buffer.slot, freeSt.ToString());
    }
    return ShardOf(key).Delete(key);
}

Status MetadataManager::StoreBegin(const BlockId& key, EntryPtr entry)
{
    auto idx = ShardIdx(key);
    entry->shard = static_cast<uint32_t>(idx);
    auto st = bufferManager_.Allocate(entry->size, entry->buffer);
    if (st == Status::NoSpace() && deepType_ == EvictionPolicyType::PREFIX_TAIL) {
        // Reclaim same-size blocks from the deepest prefix tail across every
        // shard until this size pool has a free slot, or no tail can be taken.
        const auto attemptLimit = std::max<std::size_t>(GetKeyCnt(), 1);
        for (std::size_t attempt = 0; attempt < attemptLimit && st == Status::NoSpace();
             ++attempt) {
            if (!EvictDeepest(entry->size)) { break; }
            st = bufferManager_.Allocate(entry->size, entry->buffer);
        }
    } else if (st == Status::NoSpace()) {
        EvictOneShard(*shards_[rand() % kShardCnt]);
        st = bufferManager_.Allocate(entry->size, entry->buffer);
        if (st == Status::NoSpace()) {
            EvictOneShard(*shards_[rand() % kShardCnt], true);
            st = bufferManager_.Allocate(entry->size, entry->buffer);
        }
    }
    if (!st.Success()) {
        UC_ERROR("StoreBegin: Allocate for size {} failed, status {}.", entry->size, st.ToString());
        if (st == Status::NoSpace()) {
            // Both the periodic and the deep eviction retries have run; still NoSpace
            // means real memory pressure.
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID(kDumpNospaceFailuresTotal), 1);
        }
        return Status::Error();
    }
    const auto bufSize = entry->size;
    const auto bufSlot = entry->buffer.slot;
    st = shards_[idx]->StoreBegin(key, std::move(entry));
    if (!st.Success()) {
        auto freeSt = bufferManager_.Free(bufSize, bufSlot);
        if (!freeSt.Success()) {
            UC_ERROR("StoreBegin: Free slot {} failed, status {}.", bufSlot, freeSt.ToString());
        }
    }
    return st;
}

void MetadataManager::PerformEvict()
{
    for (auto& shard : shards_) { EvictOneShard(*shard); }
    if (deepType_ != EvictionPolicyType::PREFIX_TAIL || defaultEvictRatio_ == 0.0) { return; }
    // Trim tails only after a size pool is already full. Doing this on every
    // GC tick while the pool still has free slots would drain the cache.
    for (const auto size : bufferManager_.PoolSizes()) {
        if (!bufferManager_.IsFull(size)) { continue; }
        EvictTailFraction(size);
    }
}

void MetadataManager::EvictOneShard(ShardMetadata& s, bool deep)
{
    auto victims = deep ? s.EvictDeep(defaultEvictRatio_) : s.EvictPeriodic(defaultEvictRatio_);
    for (const auto& entry : victims) {
        auto st = bufferManager_.Free(entry->size, entry->buffer.slot);
        if (!st.Success()) {
            UC_ERROR("EvictOneShard: Free slot {} failed, status {}.", entry->buffer.slot,
                     st.ToString());
        }
        s.Delete(entry->key);
    }
}

bool MetadataManager::EvictDeepest(std::size_t size)
{
    const auto now = std::chrono::system_clock::now();
    // A mark can lose a race with a load. Rescan a few times before giving up
    // so one busy tail block does not hide the next deepest candidate.
    constexpr int kMarkAttempts = 8;
    for (int attempt = 0; attempt < kMarkAttempts; ++attempt) {
        bool sawReady = false;
        std::uint32_t deepestReady = 0;
        EntryPtr victim;
        ShardMetadata* victimShard = nullptr;
        for (auto& shard : shards_) {
            const auto candidate = shard->FindTail(size, now);
            if (candidate.sawReady && (!sawReady || candidate.deepestReady > deepestReady)) {
                sawReady = true;
                deepestReady = candidate.deepestReady;
            }
            if (candidate.evictable &&
                (victim == nullptr || candidate.evictable->position > victim->position)) {
                victim = candidate.evictable;
                victimShard = shard.get();
            }
        }
        if (victim == nullptr || victimShard == nullptr) { return false; }
        if (sawReady && deepestReady > victim->position) { return false; }
        if (!victimShard->TryMarkForEvict(victim, now)) { continue; }
        const auto freeStatus = bufferManager_.Free(victim->size, victim->buffer.slot);
        if (!freeStatus.Success()) {
            UC_ERROR("EvictDeepest: Free slot {} failed, status {}.", victim->buffer.slot,
                     freeStatus.ToString());
        }
        victimShard->Delete(victim->key);
        return true;
    }
    return false;
}

void MetadataManager::EvictTailFraction(std::size_t size)
{
    std::size_t count = 0;
    for (const auto& shard : shards_) { count += shard->CountSize(size); }
    if (count == 0) { return; }
    const auto target = std::max<std::size_t>(
        1, static_cast<std::size_t>(static_cast<double>(count) * defaultEvictRatio_));
    for (std::size_t index = 0; index < target; ++index) {
        if (!EvictDeepest(size)) { return; }
    }
}

}  // namespace UC::DramPool
