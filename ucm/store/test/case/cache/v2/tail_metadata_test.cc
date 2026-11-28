/**
 * MIT License
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>
#include "cache/v2/tail_metadata.h"
#include "gtest/gtest.h"

namespace UC::Cache2 {
namespace {
using M = TailMetadata;
M::Id Id(unsigned char c)
{
    M::Id id{};
    id[0] = static_cast<std::byte>(c);
    return id;
}
struct Fixture {
    std::vector<std::max_align_t> memory;
    M metadata;
    Fixture(uint32_t nodes = 8, uint32_t shards = 16)
        : memory((M::Bytes(nodes, shards, 8) + sizeof(std::max_align_t) - 1) /
                 sizeof(std::max_align_t)), metadata(Initialize(nodes, shards)) {}
    void* Initialize(uint32_t nodes, uint32_t shards)
    {
        EXPECT_TRUE(M::Initialize(memory.data(), memory.size() * sizeof(std::max_align_t),
                                  nodes, shards, 8));
        return memory.data();
    }
};

TEST(TailMetadataTest, LateParentAndCascadeRespectDirectEdges)
{
    Fixture f;
    M::Ref a, b, c, candidate;
    ASSERT_EQ(f.metadata.Register(Id(3), Id(2), false, c), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(c, 30, 1), M::Result::Ok);
    ASSERT_EQ(f.metadata.Register(Id(1), {}, true, a), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(a, 10, 0), M::Result::Ok);
    // Missing B means C does not protect A.
    ASSERT_EQ(f.metadata.Select(0, 16, false, candidate), M::Result::Ok);
    EXPECT_EQ(candidate.index, a.index);
    ASSERT_EQ(f.metadata.AbortEvict(candidate), M::Result::Ok);
    ASSERT_EQ(f.metadata.Register(Id(2), Id(1), false, b), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(b, 20, 1), M::Result::Ok);
    EXPECT_EQ(f.metadata.Select(0, 16, false, candidate), M::Result::NotFound);
    ASSERT_EQ(f.metadata.Select(0, 16, true, candidate), M::Result::Ok);
    EXPECT_EQ(candidate.index, c.index);
    M::Shard shard{};
    ASSERT_EQ(f.metadata.GetShard(candidate, 0, shard), M::Result::Ok);
    EXPECT_EQ(shard.slot, 30u);
    EXPECT_EQ(shard.rank, 1u);
    ASSERT_EQ(f.metadata.CommitEvict(candidate), M::Result::Ok);
    ASSERT_EQ(f.metadata.Select(0, 16, true, candidate), M::Result::Ok);
    EXPECT_EQ(candidate.index, b.index);
    ASSERT_EQ(f.metadata.CommitEvict(candidate), M::Result::Ok);
    ASSERT_EQ(f.metadata.Select(0, 16, false, candidate), M::Result::Ok);
    EXPECT_EQ(candidate.index, a.index);
}

TEST(TailMetadataTest, EvictionBlocksChildAndGcRejectsStaleReference)
{
    Fixture f(2, 2);
    M::Ref a, b, chosen;
    ASSERT_EQ(f.metadata.Register(Id(1), {}, true, a), M::Result::Ok);
    ASSERT_EQ(f.metadata.Register(Id(2), Id(1), false, b), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(a, 10, 0), M::Result::Ok);
    ASSERT_EQ(f.metadata.Select(0, 4, false, chosen), M::Result::Ok);
    EXPECT_EQ(f.metadata.AddShard(b, 20, 0), M::Result::Busy);
    ASSERT_EQ(f.metadata.Release(a), M::Result::Ok);
    ASSERT_EQ(f.metadata.CommitEvict(chosen), M::Result::Ok);
    EXPECT_EQ(f.metadata.Release(a), M::Result::Invalid);
    M::Ref replacement;
    EXPECT_EQ(f.metadata.Register(Id(3), {}, true, replacement), M::Result::Ok);
    EXPECT_EQ(replacement.index, a.index);
    EXPECT_NE(replacement.generation, a.generation);
    EXPECT_EQ(f.metadata.GetStats().gc, 1u);
}

TEST(TailMetadataTest, BudgetConflictCycleAndSharedViews)
{
    Fixture f(2, 1);
    M second(f.memory.data());
    M::Ref a, b, unused;
    ASSERT_EQ(f.metadata.Register(Id(1), Id(2), false, a), M::Result::Ok);
    EXPECT_EQ(second.Register(Id(2), Id(1), false, b), M::Result::Conflict);
    ASSERT_EQ(second.Register(Id(2), {}, true, b), M::Result::Ok);
    EXPECT_EQ(second.Register(Id(3), {}, true, unused), M::Result::Full);
    EXPECT_EQ(second.Register(Id(1), {}, true, unused), M::Result::Conflict);
    EXPECT_EQ(f.metadata.AddShard(a, 1, 0), M::Result::Ok);
    EXPECT_EQ(second.AddShard(b, 2, 0), M::Result::Full);
    EXPECT_EQ(second.GetStats().failed, 2u);
}

TEST(TailMetadataTest, RemovingLastShardCreatesGapAndReleasesSpace)
{
    Fixture f(3, 3);
    M::Ref a, b, c, candidate;
    ASSERT_EQ(f.metadata.Register(Id(1), {}, true, a), M::Result::Ok);
    ASSERT_EQ(f.metadata.Register(Id(2), Id(1), false, b), M::Result::Ok);
    ASSERT_EQ(f.metadata.Register(Id(3), Id(2), false, c), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(a, 10, 0), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(b, 20, 1), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(c, 30, 1), M::Result::Ok);
    EXPECT_EQ(f.metadata.Select(0, 6, false, candidate), M::Result::NotFound);
    ASSERT_EQ(f.metadata.RemoveShard(b, 20), M::Result::Ok);
    EXPECT_EQ(f.metadata.GetStats().shards, 2u);
    ASSERT_EQ(f.metadata.Select(0, 6, false, candidate), M::Result::Ok);
    EXPECT_EQ(candidate.index, a.index);
    ASSERT_EQ(f.metadata.AbortEvict(candidate), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(b, 21, 1), M::Result::Ok);
    EXPECT_EQ(f.metadata.Select(0, 6, false, candidate), M::Result::NotFound);
}

TEST(TailMetadataTest, StaleEvictionAttemptCannotControlNextAttempt)
{
    Fixture f(1, 1);
    M::Ref node, first, second;
    ASSERT_EQ(f.metadata.Register(Id(1), {}, true, node), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(node, 10, 0), M::Result::Ok);
    ASSERT_EQ(f.metadata.Select(0, 2, false, first), M::Result::Ok);
    ASSERT_EQ(f.metadata.AbortEvict(first), M::Result::Ok);
    ASSERT_EQ(f.metadata.Select(0, 1, false, second), M::Result::Ok);
    EXPECT_EQ(first.index, second.index);
    EXPECT_EQ(first.generation, second.generation);
    EXPECT_NE(first.ticket, second.ticket);
    M::Shard shard{};
    EXPECT_EQ(f.metadata.GetShard(first, 0, shard), M::Result::Invalid);
    EXPECT_EQ(f.metadata.CommitEvict(first), M::Result::Invalid);
    EXPECT_EQ(f.metadata.AbortEvict(first), M::Result::Invalid);
    EXPECT_EQ(f.metadata.GetShard(second, 0, shard), M::Result::Ok);
    EXPECT_EQ(f.metadata.CommitEvict(second), M::Result::Ok);
    EXPECT_EQ(f.metadata.GetStats().shards, 0u);
}

TEST(TailMetadataTest, SnapshotCopiesAllShardsOnlyForCurrentEviction)
{
    Fixture f(1, 3);
    M::Ref node, selected;
    ASSERT_EQ(f.metadata.Register(Id(1), {}, true, node), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(node, 10, 0), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(node, 20, 1), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(node, 30, 2), M::Result::Ok);
    ASSERT_EQ(f.metadata.Select(0, 2, false, selected), M::Result::Ok);
    std::vector<M::Shard> shards;
    ASSERT_EQ(f.metadata.GetShards(selected, shards), M::Result::Ok);
    ASSERT_EQ(shards.size(), 3u);
    EXPECT_EQ(shards[0].slot, 30u);
    EXPECT_EQ(shards[1].slot, 20u);
    EXPECT_EQ(shards[2].slot, 10u);
    EXPECT_EQ(f.metadata.AbortEvict(selected), M::Result::Ok);
    EXPECT_EQ(f.metadata.GetShards(selected, shards), M::Result::Invalid);
}

TEST(TailMetadataTest, ConcurrentChildAdmissionAndParentSelectionAreExclusive)
{
    for (int iteration = 0; iteration < 128; ++iteration) {
        Fixture f(2, 2);
        M::Ref parent, child, selected;
        ASSERT_EQ(f.metadata.Register(Id(1), {}, true, parent), M::Result::Ok);
        ASSERT_EQ(f.metadata.Register(Id(2), Id(1), false, child), M::Result::Ok);
        ASSERT_EQ(f.metadata.AddShard(parent, 10, 0), M::Result::Ok);
        // Clear the parent's CLOCK bit before racing admission with selection.
        ASSERT_EQ(f.metadata.Select(0, 4, false, selected), M::Result::Ok);
        ASSERT_EQ(f.metadata.AbortEvict(selected), M::Result::Ok);

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        M::Result admission = M::Result::Invalid;
        M::Result selection = M::Result::Invalid;
        std::thread admit([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            admission = f.metadata.AddShard(child, 20, 0);
        });
        std::thread evict([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            selection = f.metadata.Select(0, 4, false, selected);
        });
        while (ready.load(std::memory_order_acquire) != 2) { std::this_thread::yield(); }
        go.store(true, std::memory_order_release);
        admit.join();
        evict.join();

        if (selection == M::Result::Ok && selected.index == parent.index) {
            EXPECT_EQ(admission, M::Result::Busy);
            EXPECT_EQ(f.metadata.GetStats().shards, 1u);
            ASSERT_EQ(f.metadata.AbortEvict(selected), M::Result::Ok);
        } else {
            EXPECT_EQ(admission, M::Result::Ok);
            EXPECT_EQ(f.metadata.GetStats().shards, 2u);
            if (selection == M::Result::Ok) {
                EXPECT_EQ(selected.index, child.index);
                ASSERT_EQ(f.metadata.AbortEvict(selected), M::Result::Ok);
            } else {
                EXPECT_EQ(selection, M::Result::NotFound);
            }
        }
    }
}
} // namespace
} // namespace UC::Cache2
