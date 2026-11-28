/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache/v2/cache_buffer.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <new>
#include <optional>
#include <set>
#include <thread>
#include <type_traits>
#include <vector>

namespace UC::Cache2 {

struct BufferTestAccess {
    static size_t TotalSize(size_t bucketCount, size_t lockCount, size_t slotCount)
    {
        return CtrlLayout::TotalSize(bucketCount, lockCount, slotCount);
    }

    static void Init(Buffer& buffer, void* memory, size_t rankCount, size_t slotsPerRank,
                     size_t bucketCount, size_t lockCount)
    {
        auto& layout = buffer.ctrl_.Layout();
        layout.Bind(memory, rankCount, slotsPerRank, bucketCount, lockCount);
        layout.InitHeader(4096);
        layout.InitSlotRange(0);
        buffer.myRank_ = 0;
        buffer.rankCount_ = rankCount;
        buffer.slotsPerRank_ = slotsPerRank;
        buffer.slotSize_ = 4096;
        buffer.shardsPerBlock_ = 1;
        buffer.bucketCount_ = bucketCount;
        buffer.reservedSlots_ = 0;
    }

    static size_t ReferenceCount(Buffer& buffer, const Detail::BlockId& blockId, size_t offset)
    {
        auto iBucket = buffer.HashKey(blockId, offset);
        auto& layout = buffer.ctrl_.Layout();
        auto iNode = buffer.Lookup(layout, iBucket, blockId, offset);
        return iNode == kInvalid
                   ? kInvalid
                   : layout.SlotMetaArr()[iNode].reference.load(std::memory_order_acquire);
    }

    static void InitRank(Buffer& buffer, size_t rank)
    {
        buffer.ctrl_.Layout().InitSlotRange(rank);
        buffer.myRank_ = rank;
    }

    static void Attach(Buffer& buffer, void* memory, size_t rankCount, size_t slotsPerRank,
                       size_t bucketCount, size_t lockCount, size_t rank)
    {
        auto& layout = buffer.ctrl_.Layout();
        layout.Bind(memory, rankCount, slotsPerRank, bucketCount, lockCount);
        layout.InitSlotRange(rank);
        buffer.myRank_ = rank;
        buffer.rankCount_ = rankCount;
        buffer.slotsPerRank_ = slotsPerRank;
        buffer.slotSize_ = 4096;
        buffer.shardsPerBlock_ = 1;
        buffer.bucketCount_ = bucketCount;
        buffer.reservedSlots_ = 0;
    }

    static size_t TryAcquireSlot(Buffer& buffer, const Detail::BlockId& blockId, size_t offset,
                                 size_t attempts, bool& owner, size_t preferredRank = kInvalid)
    {
        return buffer.TryAcquireSlot(blockId, offset, false, attempts, owner, preferredRank);
    }

    static void Release(Buffer& buffer, size_t slot) { buffer.Release(slot); }

    static void MarkReady(Buffer& buffer, size_t slot) { buffer.MarkReady(slot); }

    static CtrlLayout& Layout(Buffer& buffer) { return buffer.ctrl_.Layout(); }

    static size_t UsedSlots(Buffer& buffer)
    {
        return buffer.ctrl_.Layout().Hdr()->usedSlots.load(std::memory_order_relaxed);
    }

    static void ReportCapacityStats(Buffer& buffer) { buffer.ReportCapacityStats(); }

    static void RegisterMetricsDomain(Buffer& buffer, const std::string& id)
    {
        buffer.metricsDomainId_ = id;
        Buffer::Reporter().Add(&buffer);
        buffer.metricsRegistered_ = true;
    }

    static void UnregisterMetricsDomain(Buffer& buffer)
    {
        Buffer::Reporter().Remove(&buffer);
        buffer.metricsRegistered_ = false;
    }

    static std::pair<double, double> CapacitySnapshot() { return Buffer::Reporter().Snapshot(); }

    static size_t BucketOf(Buffer& buffer, const Detail::BlockId& blockId, size_t offset = 0)
    {
        return buffer.HashKey(blockId, offset);
    }

    static void SetShardsPerBlock(Buffer& buffer, size_t count) { buffer.shardsPerBlock_ = count; }

    static void SetTail(Buffer& buffer, void* memory)
    {
        buffer.tail_ = std::make_unique<TailMetadata>(memory);
    }

    static size_t FindSlot(Buffer& buffer, const Detail::BlockId& blockId, size_t offset)
    {
        auto iBucket = buffer.HashKey(blockId, offset);
        return buffer.Lookup(buffer.ctrl_.Layout(), iBucket, blockId, offset);
    }
};

namespace {

static_assert(!std::is_default_constructible_v<Buffer::Handle>);
static_assert(!std::is_constructible_v<Buffer::Handle, Buffer*, size_t, bool>);
static_assert(!std::is_constructible_v<bool, Buffer::Handle>);
static_assert(!std::is_copy_constructible_v<Buffer::Handle>);
static_assert(std::is_nothrow_move_constructible_v<Buffer::Handle>);
static_assert(std::is_nothrow_move_assignable_v<Buffer::Handle>);

Detail::BlockId MakeBlockId(uint32_t value)
{
    Detail::BlockId block;
    block.fill(static_cast<std::byte>(0));
    std::memcpy(block.data(), &value, sizeof(value));
    return block;
}

class Cache2BufferTest : public testing::Test {
protected:
    static constexpr size_t kRanks{1};
    static constexpr size_t kSlotsPerRank{128};
    static constexpr size_t kBuckets{128};
    static constexpr size_t kLocks{64};

    size_t bytes_{BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks* kSlotsPerRank)};
    void* memory_{::operator new(bytes_, std::align_val_t{64})};
    Buffer buffer_;

    void SetUp() override
    {
        BufferTestAccess::Init(buffer_, memory_, kRanks, kSlotsPerRank, kBuckets, kLocks);
    }

    void TearDown() override { ::operator delete(memory_, std::align_val_t{64}); }
};

TEST_F(Cache2BufferTest, GetUntilReturnsPinnedHandleOnSuccess)
{
    std::atomic_bool cancelled{false};
    const auto block = MakeBlockId(10000);
    auto result = buffer_.GetUntil(block, 0, std::chrono::steady_clock::now() +
                                               std::chrono::seconds(1), cancelled);
    ASSERT_TRUE(result.HasValue());
    auto handle = std::move(result).Value();
    EXPECT_TRUE(handle.Owner());
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 1u);
    handle.MarkReady();
    auto hit = buffer_.GetUntil(block, 0, std::chrono::steady_clock::now() +
                                            std::chrono::seconds(1), cancelled);
    ASSERT_TRUE(hit.HasValue());
    EXPECT_FALSE(hit.Value().Owner());
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 2u);
}

TEST_F(Cache2BufferTest, QuarantineKeepsPinAndRejectsSubsequentHits)
{
    const auto block = MakeBlockId(10002);
    {
        auto handle = buffer_.Get(block, 0);
        EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 1u);
        handle.Quarantine();
        handle.Quarantine();
    }
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 1u);
    EXPECT_FALSE(buffer_.Exist(block, 0));
    std::atomic_bool cancelled{false};
    auto hit = buffer_.GetUntil(block, 0, std::chrono::steady_clock::now() +
                                           std::chrono::milliseconds(10), cancelled);
    EXPECT_FALSE(hit.HasValue());
    EXPECT_EQ(hit.Error(), Status::Timeout());
}

TEST_F(Cache2BufferTest, WholeBlockClaimRollsBackPinsAndFailedCommit)
{
    Metrics::SetUp();
    Metrics::CreateStats("cache_dram_evicted_bytes_total", "counter");
    Metrics::GetAllStatsAndClear();
    const auto block = MakeBlockId(10003);
    auto& layout = BufferTestAccess::Layout(buffer_);
    auto* firstLock = layout.LockOf(BufferTestAccess::BucketOf(buffer_, block, 0));
    size_t otherOffset = 1;
    while (otherOffset < 10000 &&
           layout.LockOf(BufferTestAccess::BucketOf(buffer_, block, otherOffset)) != firstLock) {
        ++otherOffset;
    }
    ASSERT_LT(otherOffset, 10000u);
    size_t firstSlot;
    {
        auto first = buffer_.Get(block, 0);
        firstSlot = first.SlotIndex();
        first.MarkReady();
    }
    auto second = buffer_.Get(block, otherOffset);
    const auto secondSlot = second.SlotIndex();
    second.MarkReady();
    std::optional<Buffer::Handle> pin(std::move(second));
    const std::vector<Buffer::EvictionSlot> slots{{secondSlot, otherOffset}, {firstSlot, 0}};
    bool called = false;
    auto commit = [&] { called = true; return Status::OK(); };
    EXPECT_EQ(buffer_.TryEvictSlots(block, slots, commit), Status::Retry());
    EXPECT_FALSE(called);
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 0u);
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, otherOffset), 1u);
    EXPECT_EQ(BufferTestAccess::UsedSlots(buffer_), 2u);

    pin.reset();
    EXPECT_EQ(buffer_.TryEvictSlots(block, {{firstSlot, otherOffset}, {secondSlot, 0}}, commit),
              Status::Retry());
    EXPECT_FALSE(called);
    EXPECT_EQ(buffer_.TryEvictSlots(block, slots, [] { return Status::NoSpace(); }),
              Status::NoSpace());
    EXPECT_TRUE(buffer_.Exist(block, 0));
    EXPECT_TRUE(buffer_.Exist(block, otherOffset));
    EXPECT_EQ(BufferTestAccess::UsedSlots(buffer_), 2u);
    auto before = Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(before).at("cache_dram_evicted_bytes_total"), 0);

    EXPECT_TRUE(buffer_.TryEvictSlots(block, slots, commit).Success());
    EXPECT_TRUE(called);
    EXPECT_FALSE(buffer_.Exist(block, 0));
    EXPECT_FALSE(buffer_.Exist(block, otherOffset));
    EXPECT_EQ(BufferTestAccess::UsedSlots(buffer_), 0u);
    auto after = Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(after).at("cache_dram_evicted_bytes_total"), 8192);
}

TEST(Cache2TailBufferTest, PinnedLeafBlocksEvictionUntilReleased)
{
    constexpr size_t slots = 2, buckets = 8, locks = 4;
    const auto controlBytes = BufferTestAccess::TotalSize(buckets, locks, slots);
    auto* control = ::operator new(controlBytes, std::align_val_t{64});
    const auto tailBytes = TailMetadata::Bytes(8, 8, 8);
    auto* tail = ::operator new(tailBytes, std::align_val_t{64});
    ASSERT_TRUE(TailMetadata::Initialize(tail, tailBytes, 8, 8, 8));
    {
        Buffer buffer;
        BufferTestAccess::Init(buffer, control, 1, slots, buckets, locks);
        BufferTestAccess::SetTail(buffer, tail);
        const auto a = MakeBlockId(1), b = MakeBlockId(2), c = MakeBlockId(3);
        Detail::TaskDesc desc;
        desc.push_back({a, 0, {}});
        desc.push_back({b, 0, {}});
        desc.push_back({c, 0, {}});
        desc.blockRelations = {{a, {}, true}, {b, a, false}, {c, b, false}};
        auto relations = buffer.RegisterRelations(desc);
        ASSERT_TRUE(relations.HasValue());
        auto token = std::move(relations).Value();
        (void)token;
        std::atomic_bool cancelled{false};
        const auto get = [&](const Detail::BlockId& id, size_t ms) {
            return buffer.GetUntil(id, 0, std::chrono::steady_clock::now() +
                                            std::chrono::milliseconds(ms), cancelled);
        };
        {
            auto first = get(a, 1000);
            ASSERT_TRUE(first.HasValue());
            first.Value().MarkReady();
        }
        auto second = get(b, 1000);
        ASSERT_TRUE(second.HasValue());
        std::optional<Buffer::Handle> pinned(std::move(second).Value());
        pinned->MarkReady();
        auto blocked = get(c, 20);
        ASSERT_FALSE(blocked.HasValue());
        EXPECT_EQ(blocked.Error(), Status::Timeout());
        EXPECT_TRUE(buffer.Exist(a, 0));
        EXPECT_TRUE(buffer.Exist(b, 0));
        pinned.reset();
        auto inserted = get(c, 1000);
        ASSERT_TRUE(inserted.HasValue());
        inserted.Value().MarkReady();
        EXPECT_TRUE(buffer.Exist(a, 0));
        EXPECT_FALSE(buffer.Exist(b, 0));
        EXPECT_TRUE(buffer.Exist(c, 0));
        EXPECT_EQ(BufferTestAccess::UsedSlots(buffer), 2u);
    }
    ::operator delete(tail, std::align_val_t{64});
    ::operator delete(control, std::align_val_t{64});
}

TEST(Cache2TailBufferTest, RegistrationFailureRollsBackAndMissingRelationsAreRejected)
{
    constexpr size_t slots = 2, buckets = 8, locks = 4;
    const auto controlBytes = BufferTestAccess::TotalSize(buckets, locks, slots);
    auto* control = ::operator new(controlBytes, std::align_val_t{64});
    const auto tailBytes = TailMetadata::Bytes(1, 2, 2);
    auto* tail = ::operator new(tailBytes, std::align_val_t{64});
    ASSERT_TRUE(TailMetadata::Initialize(tail, tailBytes, 1, 2, 2));
    {
        Buffer buffer;
        BufferTestAccess::Init(buffer, control, 1, slots, buckets, locks);
        BufferTestAccess::SetTail(buffer, tail);
        const auto a = MakeBlockId(10), b = MakeBlockId(11);
        Detail::TaskDesc desc;
        desc.push_back({a, 0, {}});
        auto missing = buffer.RegisterRelations(desc);
        ASSERT_FALSE(missing.HasValue());
        EXPECT_EQ(missing.Error(), Status::InvalidParam());
        desc.blockRelations = {{a, {}, true}, {b, a, false}};
        auto full = buffer.RegisterRelations(desc);
        ASSERT_FALSE(full.HasValue());
        EXPECT_EQ(full.Error(), Status::NoSpace());
        TailMetadata view(tail);
        EXPECT_EQ(view.GetStats().nodes, 0u);
    }
    ::operator delete(tail, std::align_val_t{64});
    ::operator delete(control, std::align_val_t{64});
}

TEST_F(Cache2BufferTest, UsedSlotsCountsReservationsWithoutDoubleCountingHits)
{
    Metrics::SetUp();
    Metrics::CreateStats("cache_dram_capacity_bytes", "gauge");
    Metrics::CreateStats("cache_dram_used_bytes", "gauge");
    Metrics::GetAllStatsAndClear();
    EXPECT_EQ(BufferTestAccess::UsedSlots(buffer_), 0u);
    const auto block = MakeBlockId(10500);
    buffer_.Prealloc(block, 0);
    EXPECT_EQ(BufferTestAccess::UsedSlots(buffer_), 1u);
    {
        auto first = buffer_.Get(block, 0);
        auto hit = buffer_.Get(block, 0);
        EXPECT_EQ(BufferTestAccess::UsedSlots(buffer_), 1u);
        first.MarkFailed();
    }
    EXPECT_EQ(BufferTestAccess::UsedSlots(buffer_), 1u);
    BufferTestAccess::ReportCapacityStats(buffer_);
    auto stats = Metrics::GetAllStatsAndClear();
    const auto& gauges = std::get<1>(stats);
    ASSERT_EQ(gauges.count("cache_dram_capacity_bytes"), 1u);
    ASSERT_EQ(gauges.count("cache_dram_used_bytes"), 1u);
    EXPECT_EQ(gauges.at("cache_dram_capacity_bytes"), kSlotsPerRank * 4096);
    EXPECT_EQ(gauges.at("cache_dram_used_bytes"), 4096);
}

TEST(Cache2BufferMetricsTest, EvictsOnlyReadySlotsIncludingSameBucketReplacement)
{
    Metrics::SetUp();
    Metrics::CreateStats("cache_dram_evicted_bytes_total", "counter");
    Metrics::GetAllStatsAndClear();
    constexpr size_t kRanks{1}, kSlotsPerRank{1}, kBuckets{1}, kLocks{1};
    auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    {
        Buffer buffer;
        BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);
        buffer.Prealloc(MakeBlockId(20000), 0);  // Loading without an owner.
        {
            auto failed = buffer.Get(MakeBlockId(20001), 0);
            failed.MarkFailed();
        }
        {
            auto ready = buffer.Get(MakeBlockId(20002), 0);
            ready.MarkReady();
        }
        {
            auto replacement = buffer.Get(MakeBlockId(20003), 0);
            replacement.MarkReady();
        }
        auto stats = Metrics::GetAllStatsAndClear();
        const auto& counters = std::get<0>(stats);
        ASSERT_EQ(counters.count("cache_dram_evicted_bytes_total"), 1u);
        EXPECT_EQ(counters.at("cache_dram_evicted_bytes_total"), 4096);
        EXPECT_EQ(BufferTestAccess::UsedSlots(buffer), 1u);
    }
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferMetricsTest, ReporterSumsDomainsAndDeduplicatesSharedMappings)
{
    constexpr size_t kRanks{2}, kSlotsPerRank{2}, kBuckets{8}, kLocks{4};
    const auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* firstMemory = ::operator new(bytes, std::align_val_t{64});
    auto* secondMemory = ::operator new(bytes, std::align_val_t{64});
    {
        Buffer first;
        Buffer duplicate;
        Buffer second;
        BufferTestAccess::Init(first, firstMemory, kRanks, kSlotsPerRank, kBuckets, kLocks);
        BufferTestAccess::Attach(duplicate, firstMemory, kRanks, kSlotsPerRank, kBuckets, kLocks, 1);
        BufferTestAccess::Init(second, secondMemory, kRanks, kSlotsPerRank, kBuckets, kLocks);
        first.Prealloc(MakeBlockId(21000), 0);
        second.Prealloc(MakeBlockId(21001), 0);
        BufferTestAccess::RegisterMetricsDomain(first, "first");
        BufferTestAccess::RegisterMetricsDomain(duplicate, "first");
        BufferTestAccess::RegisterMetricsDomain(second, "second");
        EXPECT_EQ(BufferTestAccess::CapacitySnapshot(),
                  std::make_pair(2.0 * kRanks * kSlotsPerRank * 4096, 2.0 * 4096));
        BufferTestAccess::UnregisterMetricsDomain(first);
        EXPECT_EQ(BufferTestAccess::CapacitySnapshot().first, 2.0 * kRanks * kSlotsPerRank * 4096);
        BufferTestAccess::UnregisterMetricsDomain(duplicate);
        EXPECT_EQ(BufferTestAccess::CapacitySnapshot(),
                  std::make_pair(1.0 * kRanks * kSlotsPerRank * 4096, 1.0 * 4096));
    }
    ::operator delete(firstMemory, std::align_val_t{64});
    ::operator delete(secondMemory, std::align_val_t{64});
}

TEST_F(Cache2BufferTest, GetUntilRejectsPrecancelAndExpiredDeadlineWithoutPinning)
{
    const auto block = MakeBlockId(10001);
    auto handle = buffer_.Get(block, 0);
    handle.MarkReady();
    std::atomic_bool cancelled{true};
    auto cancelledResult = buffer_.GetUntil(block, 0, std::chrono::steady_clock::time_point::max(),
                                            cancelled);
    ASSERT_FALSE(cancelledResult.HasValue());
    EXPECT_EQ(cancelledResult.Error(), Status::Retry());
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 1u);
    cancelled.store(false);
    auto timeoutResult = buffer_.GetUntil(block, 0, std::chrono::steady_clock::now(), cancelled);
    ASSERT_FALSE(timeoutResult.HasValue());
    EXPECT_EQ(timeoutResult.Error(), Status::Timeout());
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 1u);
}

TEST_F(Cache2BufferTest, GetUntilCanCancelOrTimeoutWhileAllSlotsArePinned)
{
    std::vector<Buffer::Handle> held;
    held.reserve(kSlotsPerRank);
    for (size_t i = 0; i < kSlotsPerRank; ++i) {
        held.push_back(buffer_.Get(MakeBlockId(static_cast<uint32_t>(11000 + i)), 0));
        held.back().MarkReady();
    }
    std::atomic_bool cancelled{false};
    const auto missing = MakeBlockId(12000);
    auto timeout = buffer_.GetUntil(missing, 0, std::chrono::steady_clock::now() +
                                                   std::chrono::milliseconds(5), cancelled);
    ASSERT_FALSE(timeout.HasValue());
    EXPECT_EQ(timeout.Error(), Status::Timeout());

    std::promise<void> started;
    auto startedFuture = started.get_future();
    auto pending = std::async(std::launch::async, [&] {
        started.set_value();
        return buffer_.GetUntil(missing, 0, std::chrono::steady_clock::now() +
                                                 std::chrono::seconds(2), cancelled);
    });
    startedFuture.wait();
    cancelled.store(true, std::memory_order_release);
    auto result = pending.get();
    ASSERT_FALSE(result.HasValue());
    EXPECT_EQ(result.Error(), Status::Retry());
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, MakeBlockId(11000), 0), 1u);
    EXPECT_FALSE(buffer_.Exist(missing, 0));
}

TEST_F(Cache2BufferTest, LayerOffsetsUseDevelopHashMixAndSeparateBuckets)
{
    constexpr size_t kLayers{78};
    const auto block = MakeBlockId(78);
    const auto h1 = Detail::BlockIdHasher{}(block);
    constexpr auto goldenSection = 0x9e3779b97f4a7c15ULL;
    std::set<size_t> buckets;
    for (size_t layer = 0; layer < kLayers; ++layer) {
        const auto h2 = std::hash<size_t>{}(layer);
        const auto expected = (h1 ^ (h2 + goldenSection + (h1 << 6) + (h1 >> 2))) % kBuckets;
        const auto bucket = BufferTestAccess::BucketOf(buffer_, block, layer);
        EXPECT_EQ(bucket, expected) << "layer " << layer;
        buckets.insert(bucket);
    }
    EXPECT_EQ(buckets.size(), kLayers);
}

TEST_F(Cache2BufferTest, LayerwisePreallocAndHitsRetainDistinctSlots)
{
    constexpr size_t kLayers{78};
    const auto block = MakeBlockId(79);
    std::vector<size_t> slots;
    for (size_t layer = 0; layer < kLayers; ++layer) {
        buffer_.Prealloc(block, layer);
        ASSERT_TRUE(buffer_.Exist(block, layer));
        EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, layer), 0);
        auto handle = buffer_.Get(block, layer);
        EXPECT_TRUE(handle.Owner());
        handle.MarkReady();
        slots.push_back(handle.SlotIndex());
        EXPECT_EQ(BufferTestAccess::Layout(buffer_).SlotMetaArr()[handle.SlotIndex()].hash.load(),
                  BufferTestAccess::BucketOf(buffer_, block, layer));
    }
    const std::set<size_t> distinctSlots(slots.begin(), slots.end());
    EXPECT_EQ(distinctSlots.size(), kLayers);
    EXPECT_FALSE(buffer_.Exist(block, kLayers));
    for (size_t layer = 0; layer < kLayers; ++layer) {
        buffer_.Prealloc(block, layer);
        auto handle = buffer_.Get(block, layer);
        EXPECT_FALSE(handle.Owner());
        EXPECT_EQ(handle.GetState(), CtrlLayout::SlotMeta::State::Ready);
        EXPECT_EQ(handle.SlotIndex(), slots[layer]);
    }
}

TEST_F(Cache2BufferTest, TouchMarksAllLayerBucketsWithoutTouchingAnotherBlock)
{
    constexpr size_t kLayers{78};
    BufferTestAccess::SetShardsPerBlock(buffer_, kLayers);
    const auto block = MakeBlockId(80);
    const auto otherBlock = MakeBlockId(81);
    std::vector<size_t> slots;
    for (size_t layer = 0; layer < kLayers; ++layer) {
        auto handle = buffer_.Get(block, layer);
        handle.MarkReady();
        slots.push_back(handle.SlotIndex());
    }
    size_t otherSlot;
    {
        auto handle = buffer_.Get(otherBlock, 0);
        handle.MarkReady();
        otherSlot = handle.SlotIndex();
    }
    auto* meta = BufferTestAccess::Layout(buffer_).SlotMetaArr();
    for (const auto slot : slots) { meta[slot].accessed.store(0); }
    meta[otherSlot].accessed.store(0);

    buffer_.Touch(&block, 1);
    for (const auto slot : slots) {
        EXPECT_EQ(meta[slot].accessed.load(), 1);
        EXPECT_EQ(meta[slot].reference.load(), 0);
        EXPECT_EQ(meta[slot].state.load(), CtrlLayout::SlotMeta::State::Ready);
    }
    EXPECT_EQ(meta[otherSlot].accessed.load(), 0);
}

TEST_F(Cache2BufferTest, TouchHandlesMissingLayersAndEmptyInputs)
{
    BufferTestAccess::SetShardsPerBlock(buffer_, 78);
    const auto block = MakeBlockId(82);
    const auto missing = MakeBlockId(83);
    size_t slot;
    {
        auto handle = buffer_.Get(block, 37);
        handle.MarkReady();
        slot = handle.SlotIndex();
    }
    auto& meta = BufferTestAccess::Layout(buffer_).SlotMetaArr()[slot];
    meta.accessed.store(0);
    buffer_.Touch(nullptr, 1);
    buffer_.Touch(&block, 0);
    buffer_.Touch(&missing, 1);
    EXPECT_EQ(meta.accessed.load(), 0);
    buffer_.Touch(&block, 1);
    EXPECT_EQ(meta.accessed.load(), 1);
}

TEST_F(Cache2BufferTest, PreallocLeavesOwnerElectionForDemandGet)
{
    auto block = MakeBlockId(1);
    buffer_.Prealloc(block, 0);
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 0);

    auto owner = buffer_.Get(block, 0);
    EXPECT_TRUE(owner.Owner());
    EXPECT_EQ(owner.GetState(), CtrlLayout::SlotMeta::State::Loading);

    auto reader = buffer_.Get(block, 0);
    EXPECT_FALSE(reader.Owner());
    owner.MarkReady();
    EXPECT_EQ(reader.GetState(), CtrlLayout::SlotMeta::State::Ready);
}

TEST_F(Cache2BufferTest, ConcurrentPreallocAndDemandElectExactlyOneOwner)
{
    constexpr size_t kThreads{16};
    constexpr size_t kRounds{64};

    for (size_t round = 0; round < kRounds; ++round) {
        auto block = MakeBlockId(static_cast<uint32_t>(round + 100));
        std::atomic<bool> start{false};
        std::atomic<size_t> owners{0};
        std::atomic<size_t> acquired{0};
        std::thread prealloc([&] {
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            buffer_.Prealloc(block, 0);
        });
        std::vector<std::thread> loads;
        loads.reserve(kThreads);
        for (size_t i = 0; i < kThreads; ++i) {
            loads.emplace_back([&] {
                while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
                auto handle = buffer_.Get(block, 0);
                acquired.fetch_add(1, std::memory_order_relaxed);
                if (handle.Owner()) {
                    owners.fetch_add(1, std::memory_order_relaxed);
                    handle.MarkReady();
                }
            });
        }
        start.store(true, std::memory_order_release);
        prealloc.join();
        for (auto& load : loads) { load.join(); }
        EXPECT_EQ(acquired.load(std::memory_order_relaxed), kThreads) << "round " << round;
        EXPECT_EQ(owners.load(std::memory_order_relaxed), 1) << "round " << round;
    }
}

TEST_F(Cache2BufferTest, ExistDoesNotStealPreallocatedOwner)
{
    auto block = MakeBlockId(2);
    buffer_.Prealloc(block, 0);
    ASSERT_TRUE(buffer_.Exist(block, 0));
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 0);

    auto owner = buffer_.Get(block, 0);
    EXPECT_TRUE(owner.Owner());
    owner.MarkReady();
}

TEST_F(Cache2BufferTest, AbandonedOwnerPublishesFailureAndCanRetry)
{
    auto block = MakeBlockId(3);
    {
        std::optional<Buffer::Handle> reader;
        {
            auto owner = buffer_.Get(block, 0);
            reader.emplace(buffer_.Get(block, 0));
            EXPECT_TRUE(owner.Owner());
            EXPECT_FALSE(reader->Owner());
        }
        EXPECT_EQ(reader->GetState(), CtrlLayout::SlotMeta::State::Failed);
    }

    auto retry = buffer_.Get(block, 0);
    EXPECT_TRUE(retry.Owner());
    EXPECT_EQ(retry.GetState(), CtrlLayout::SlotMeta::State::Loading);
    retry.MarkReady();
    EXPECT_EQ(retry.GetState(), CtrlLayout::SlotMeta::State::Ready);
}

TEST_F(Cache2BufferTest, MoveTransfersReferenceWithoutAbandoningOwner)
{
    auto block = MakeBlockId(4);
    {
        auto source = buffer_.Get(block, 0);
        {
            auto moved = std::move(source);
            EXPECT_TRUE(moved.Owner());
            EXPECT_EQ(moved.GetState(), CtrlLayout::SlotMeta::State::Loading);
            EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 1);
            moved.MarkReady();
        }
        EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 0);
    }
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 0);
    auto reader = buffer_.Get(block, 0);
    EXPECT_FALSE(reader.Owner());
    EXPECT_EQ(reader.GetState(), CtrlLayout::SlotMeta::State::Ready);
}

TEST_F(Cache2BufferTest, MoveAssignmentAbandonsOldOwnerAndTransfersNewReference)
{
    auto oldBlock = MakeBlockId(5);
    auto newBlock = MakeBlockId(6);
    auto reader = [&] {
        auto destination = buffer_.Get(oldBlock, 0);
        auto observer = buffer_.Get(oldBlock, 0);
        auto source = buffer_.Get(newBlock, 0);
        destination = std::move(source);
        EXPECT_EQ(observer.GetState(), CtrlLayout::SlotMeta::State::Failed);
        EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, oldBlock, 0), 1);
        EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, newBlock, 0), 1);
        EXPECT_TRUE(destination.Owner());
        destination.MarkReady();
        return observer;
    }();
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, newBlock, 0), 0);
    EXPECT_EQ(reader.GetState(), CtrlLayout::SlotMeta::State::Failed);
}

TEST_F(Cache2BufferTest, GetWaitsForPinnedSlotAndReturnsAnAcquiredHandle)
{
    std::vector<Buffer::Handle> pinned;
    pinned.reserve(kSlotsPerRank);
    for (uint32_t value = 0; value < kSlotsPerRank; ++value) {
        pinned.push_back(buffer_.Get(MakeBlockId(value), 0));
        pinned.back().MarkReady();
    }

    auto releasedSlot = pinned.back().SlotIndex();
    auto block = MakeBlockId(1000);
    std::promise<void> started;
    auto entered = started.get_future();
    auto pending = std::async(std::launch::async, [&] {
        started.set_value();
        return buffer_.Get(block, 0);
    });
    entered.wait();
    EXPECT_EQ(pending.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    pinned.pop_back();

    auto handle = pending.get();
    EXPECT_EQ(handle.SlotIndex(), releasedSlot);
    EXPECT_TRUE(handle.Owner());
    EXPECT_EQ(handle.GetState(), CtrlLayout::SlotMeta::State::Loading);
    EXPECT_EQ(BufferTestAccess::ReferenceCount(buffer_, block, 0), 1);
    handle.MarkReady();
}

TEST(Cache2BufferPartitionTest, ClockEvictsOnlyInsideLocalRankAndHonorsSecondChanceAndPins)
{
    constexpr size_t kRanks{2};
    constexpr size_t kSlotsPerRank{4};
    constexpr size_t kBuckets{16};
    constexpr size_t kLocks{8};
    auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer buffer;
    BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);
    BufferTestAccess::InitRank(buffer, 1);

    {
        std::optional<Buffer::Handle> pinned;
        for (uint32_t value = 1; value <= kSlotsPerRank; ++value) {
            auto handle = buffer.Get(MakeBlockId(value), 0);
            EXPECT_GE(handle.SlotIndex(), kSlotsPerRank);
            EXPECT_LT(handle.SlotIndex(), kRanks * kSlotsPerRank);
            EXPECT_TRUE(handle.Owner());
            handle.MarkReady();
            if (value == 1) { pinned.emplace(std::move(handle)); }
        }

        auto replacement = buffer.Get(MakeBlockId(100), 0);
        EXPECT_EQ(replacement.SlotIndex(), kSlotsPerRank + 1);
        EXPECT_TRUE(replacement.Owner());
        replacement.MarkReady();
        EXPECT_FALSE(buffer.Exist(MakeBlockId(2), 0));

        ASSERT_TRUE(pinned.has_value());
        EXPECT_EQ(pinned->SlotIndex(), kSlotsPerRank);
        EXPECT_EQ(pinned->GetState(), CtrlLayout::SlotMeta::State::Ready);
        EXPECT_TRUE(buffer.Exist(MakeBlockId(1), 0));
        for (size_t i = 0; i < kSlotsPerRank; ++i) {
            auto& meta = BufferTestAccess::Layout(buffer).SlotMetaArr()[i];
            EXPECT_EQ(meta.reference.load(std::memory_order_relaxed), 0);
            EXPECT_EQ(meta.hash.load(std::memory_order_relaxed), kInvalid);
        }
    }
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferPartitionTest, PreferredRankClaimsPreallocatedKeyInItsOwnSegment)
{
    constexpr size_t kRanks{2};
    constexpr size_t kSlotsPerRank{4};
    constexpr size_t kBuckets{16};
    constexpr size_t kLocks{8};
    auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer rank0;
    Buffer rank1;
    BufferTestAccess::Init(rank0, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);
    BufferTestAccess::Attach(rank1, memory, kRanks, kSlotsPerRank, kBuckets, kLocks, 1);

    auto block = MakeBlockId(1000);
    rank1.Prealloc(block, 0, false, 1);
    EXPECT_EQ(BufferTestAccess::UsedSlots(rank0), 1u);
    EXPECT_EQ(BufferTestAccess::UsedSlots(rank1), 1u);

    bool owner = false;
    auto wrongRank = BufferTestAccess::TryAcquireSlot(rank0, block, 0, 1, owner, 1);
    EXPECT_EQ(wrongRank, kInvalid);

    auto slot = BufferTestAccess::TryAcquireSlot(rank1, block, 0, 1, owner, 1);
    ASSERT_NE(slot, kInvalid);
    EXPECT_TRUE(owner);
    EXPECT_GE(slot, kSlotsPerRank);
    EXPECT_LT(slot, kRanks * kSlotsPerRank);
    BufferTestAccess::MarkReady(rank1, slot);

    auto reader = BufferTestAccess::TryAcquireSlot(rank0, block, 0, 1, owner, 1);
    ASSERT_EQ(reader, slot);
    EXPECT_FALSE(owner);
    BufferTestAccess::Release(rank0, reader);
    BufferTestAccess::Release(rank1, slot);
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferPartitionTest, PreallocRunsOnlyOnPreferredRank)
{
    constexpr size_t kRanks{2};
    constexpr size_t kSlotsPerRank{4};
    constexpr size_t kBuckets{16};
    constexpr size_t kLocks{8};
    auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer rank0;
    Buffer rank1;
    BufferTestAccess::Init(rank0, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);
    BufferTestAccess::Attach(rank1, memory, kRanks, kSlotsPerRank, kBuckets, kLocks, 1);

    auto block = MakeBlockId(1001);
    rank0.Prealloc(block, 0, false, 1);
    EXPECT_EQ(BufferTestAccess::FindSlot(rank0, block, 0), kInvalid);
    rank1.Prealloc(block, 0, false, 1);
    const auto slot = BufferTestAccess::FindSlot(rank0, block, 0);
    EXPECT_GE(slot, kSlotsPerRank);
    EXPECT_LT(slot, kRanks * kSlotsPerRank);
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferPartitionTest, NonPreferredRankCanObserveFailedSlotWithoutTakingOwnership)
{
    constexpr size_t kRanks{2};
    constexpr size_t kSlotsPerRank{4};
    constexpr size_t kBuckets{16};
    constexpr size_t kLocks{8};
    auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer rank0;
    Buffer rank1;
    BufferTestAccess::Init(rank0, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);
    BufferTestAccess::Attach(rank1, memory, kRanks, kSlotsPerRank, kBuckets, kLocks, 1);

    auto block = MakeBlockId(1002);
    {
        auto failedOwner = rank1.Get(block, 0, false, 1);
        ASSERT_TRUE(failedOwner.Owner());
        failedOwner.MarkFailed();
    }

    bool owner = true;
    auto slot = BufferTestAccess::TryAcquireSlot(rank0, block, 0, 1, owner, 1);
    ASSERT_NE(slot, kInvalid);
    EXPECT_FALSE(owner);
    EXPECT_EQ(BufferTestAccess::Layout(rank0).SlotMetaArr()[slot].state.load(),
              CtrlLayout::SlotMeta::State::Failed);
    BufferTestAccess::Release(rank0, slot);
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferLockTest, CrossStripeMigrationRollsBackAndCanRetry)
{
    Metrics::SetUp();
    Metrics::CreateStats("cache_dram_evicted_bytes_total", "counter");
    Metrics::GetAllStatsAndClear();
    constexpr size_t kRanks{1};
    constexpr size_t kSlotsPerRank{1};
    constexpr size_t kBuckets{8};
    constexpr size_t kLocks{4};
    auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer buffer;
    BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);

    auto oldBlock = MakeBlockId(1);
    auto oldBucket = BufferTestAccess::BucketOf(buffer, oldBlock);
    auto& layout = BufferTestAccess::Layout(buffer);
    Detail::BlockId newBlock;
    bool foundDifferentStripe = false;
    for (uint32_t value = 2; value < 4096; ++value) {
        auto candidate = MakeBlockId(value);
        if (layout.LockOf(BufferTestAccess::BucketOf(buffer, candidate)) !=
            layout.LockOf(oldBucket)) {
            newBlock = candidate;
            foundDifferentStripe = true;
            break;
        }
    }
    ASSERT_TRUE(foundDifferentStripe);

    size_t slot;
    {
        auto old = buffer.Get(oldBlock, 0);
        old.MarkReady();
        slot = old.SlotIndex();
    }
    layout.SlotMetaArr()[slot].accessed.store(0, std::memory_order_relaxed);

    auto* oldLock = layout.LockOf(oldBucket);
    oldLock->Lock();
    bool owner = false;
    auto failed = BufferTestAccess::TryAcquireSlot(buffer, newBlock, 0, 1, owner);
    EXPECT_EQ(failed, kInvalid);
    EXPECT_EQ(layout.SlotMetaArr()[slot].reference.load(std::memory_order_acquire), 0);
    EXPECT_EQ(layout.SlotMetaArr()[slot].hash.load(std::memory_order_acquire), oldBucket);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, oldBlock, 0), slot);
    EXPECT_EQ(BufferTestAccess::UsedSlots(buffer), 1u);
    auto failedStats = Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(failedStats).count("cache_dram_evicted_bytes_total"), 0u);
    oldLock->Unlock();

    layout.SlotMetaArr()[slot].accessed.store(0, std::memory_order_relaxed);
    auto replacement = BufferTestAccess::TryAcquireSlot(buffer, newBlock, 0, 1, owner);
    ASSERT_NE(replacement, kInvalid);
    EXPECT_TRUE(owner);
    EXPECT_EQ(replacement, slot);
    BufferTestAccess::MarkReady(buffer, replacement);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, oldBlock, 0), kInvalid);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, newBlock, 0), slot);
    EXPECT_EQ(BufferTestAccess::UsedSlots(buffer), 1u);
    auto replacementStats = Metrics::GetAllStatsAndClear();
    ASSERT_EQ(std::get<0>(replacementStats).count("cache_dram_evicted_bytes_total"), 1u);
    EXPECT_EQ(std::get<0>(replacementStats).at("cache_dram_evicted_bytes_total"), 4096);

    BufferTestAccess::Release(buffer, replacement);
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferOptimisticTest, ReadyHitSucceedsWhileBucketStripeIsLocked)
{
    constexpr size_t kRanks{1};
    constexpr size_t kSlotsPerRank{4};
    constexpr size_t kBuckets{8};
    constexpr size_t kLocks{4};
    auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer buffer;
    BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);

    auto block = MakeBlockId(200);
    size_t slot;
    {
        auto owner = buffer.Get(block, 0);
        owner.MarkReady();
        slot = owner.SlotIndex();
    }

    auto& layout = BufferTestAccess::Layout(buffer);
    auto* lock = layout.LockOf(BufferTestAccess::BucketOf(buffer, block));
    lock->Lock();
    bool owner = true;
    auto hit = BufferTestAccess::TryAcquireSlot(buffer, block, 0, 1, owner);
    lock->Unlock();
    ASSERT_NE(hit, kInvalid);
    EXPECT_FALSE(owner);
    EXPECT_EQ(hit, slot);
    EXPECT_EQ(layout.SlotMetaArr()[hit].state.load(std::memory_order_acquire),
              CtrlLayout::SlotMeta::State::Ready);

    BufferTestAccess::Release(buffer, hit);
    ::operator delete(memory, std::align_val_t{64});
}

TEST(Cache2BufferOptimisticTest, PinnedHitPreventsSlotReconfigurationUntilRelease)
{
    constexpr size_t kRanks{1};
    constexpr size_t kSlotsPerRank{1};
    constexpr size_t kBuckets{8};
    constexpr size_t kLocks{4};
    auto bytes = BufferTestAccess::TotalSize(kBuckets, kLocks, kRanks * kSlotsPerRank);
    auto* memory = ::operator new(bytes, std::align_val_t{64});
    Buffer buffer;
    BufferTestAccess::Init(buffer, memory, kRanks, kSlotsPerRank, kBuckets, kLocks);

    auto oldBlock = MakeBlockId(300);
    auto newBlock = MakeBlockId(301);
    {
        auto owner = buffer.Get(oldBlock, 0);
        owner.MarkReady();
    }

    bool owner = true;
    auto pinned = BufferTestAccess::TryAcquireSlot(buffer, oldBlock, 0, 1, owner);
    ASSERT_NE(pinned, kInvalid);
    EXPECT_FALSE(owner);
    auto& meta = BufferTestAccess::Layout(buffer).SlotMetaArr()[pinned];
    meta.accessed.store(0, std::memory_order_relaxed);
    auto blocked = BufferTestAccess::TryAcquireSlot(buffer, newBlock, 0, 2, owner);
    EXPECT_EQ(blocked, kInvalid);
    EXPECT_NE(BufferTestAccess::FindSlot(buffer, oldBlock, 0), kInvalid);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, newBlock, 0), kInvalid);

    auto slot = pinned;
    BufferTestAccess::Release(buffer, pinned);
    meta.accessed.store(0, std::memory_order_relaxed);
    auto replacement = BufferTestAccess::TryAcquireSlot(buffer, newBlock, 0, 1, owner);
    ASSERT_NE(replacement, kInvalid);
    EXPECT_TRUE(owner);
    EXPECT_EQ(replacement, slot);
    BufferTestAccess::MarkReady(buffer, replacement);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, oldBlock, 0), kInvalid);
    EXPECT_EQ(BufferTestAccess::FindSlot(buffer, newBlock, 0), slot);

    BufferTestAccess::Release(buffer, replacement);
    ::operator delete(memory, std::align_val_t{64});
}

}  // namespace
}  // namespace UC::Cache2
