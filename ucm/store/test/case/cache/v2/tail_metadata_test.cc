/**
 * MIT License
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#include <atomic>
#include <cstddef>
#include <chrono>
#include <memory>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "cache/v2/tail_metadata.h"
#include "gtest/gtest.h"

namespace UC::Cache2 {
struct TailMetadataTestAccess {
    static pthread_mutex_t* Mutex(void* memory)
    {
        return &static_cast<TailMetadata::Header*>(memory)->lock;
    }
};
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

TEST(TailMetadataTest, LogicalUsageAndGcAreSharedAndReclaimed)
{
    Fixture f(2, 2);
    M other(f.memory.data());
    const auto empty = f.metadata.GetStats();
    EXPECT_EQ(empty.capacityBytes, M::Bytes(2, 2, 8));
    EXPECT_GT(empty.usedBytes, 0u); // Header and hash buckets remain occupied.
    EXPECT_LT(empty.usedBytes, empty.capacityBytes);
    M::Ref ref;
    ASSERT_EQ(f.metadata.Register(Id(1), {}, true, ref), M::Result::Ok);
    const auto registered = other.GetStats();
    EXPECT_GT(registered.usedBytes, empty.usedBytes);
    ASSERT_EQ(other.AddShard(ref, 1, 0), M::Result::Ok);
    const auto resident = f.metadata.GetStats();
    EXPECT_GT(resident.usedBytes, registered.usedBytes);
    EXPECT_EQ(resident.capacityBytes, empty.capacityBytes);
    ASSERT_EQ(other.RemoveShard(ref, 1), M::Result::Ok);
    EXPECT_EQ(f.metadata.GetStats().usedBytes, registered.usedBytes);
    ASSERT_EQ(f.metadata.Release(ref), M::Result::Ok);
    const auto collected = other.GetStats();
    EXPECT_EQ(collected.usedBytes, empty.usedBytes);
    EXPECT_EQ(collected.gc, empty.gc + 1);
    EXPECT_EQ(collected.capacityBytes, empty.capacityBytes);
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

TEST(TailMetadataTest, MetadataReclamationRejectsRacingReferences)
{
    Fixture f(1, 1);
    M::Ref node, selected, retry;
    ASSERT_EQ(f.metadata.Register(Id(1), {}, true, node), M::Result::Ok);
    ASSERT_EQ(f.metadata.AddShard(node, 10, 1), M::Result::Ok);
    EXPECT_EQ(f.metadata.Select(0, 2, false, selected, true, true), M::Result::NotFound);
    ASSERT_EQ(f.metadata.Release(node), M::Result::Ok);
    ASSERT_EQ(f.metadata.Select(0, 2, false, selected, true, true), M::Result::Ok);
    EXPECT_EQ(f.metadata.Register(Id(1), {}, true, retry), M::Result::Busy);
    ASSERT_EQ(f.metadata.CommitEvict(selected), M::Result::Ok);
    EXPECT_EQ(f.metadata.GetStats().nodes, 0u);
    EXPECT_EQ(f.metadata.Register(Id(1), {}, true, retry), M::Result::Ok);
}

TEST(TailMetadataTest, LockContentionHasBoundedWait)
{
    Fixture f;
    auto* mutex = TailMetadataTestAccess::Mutex(f.memory.data());
    std::atomic_bool locked{false}, release{false};
    std::thread holder([&] {
        if (pthread_mutex_lock(mutex) != 0) { return; }
        locked.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) { std::this_thread::yield(); }
        (void)pthread_mutex_unlock(mutex);
    });
    const auto start = std::chrono::steady_clock::now();
    while (!locked.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
        std::this_thread::yield();
    }
    M::Ref ref;
    const auto result = locked.load() ? f.metadata.Register(Id(1), {}, true, ref)
                                      : M::Result::Invalid;
    const auto elapsed = std::chrono::steady_clock::now() - start;
    release.store(true, std::memory_order_release);
    holder.join();
    EXPECT_EQ(result, M::Result::Busy);
    EXPECT_LT(elapsed, std::chrono::seconds(6));
    EXPECT_EQ(f.metadata.Register(Id(1), {}, true, ref), M::Result::Ok);
}

TEST(TailMetadataTest, DeadOwnerPoisonsSharedDomain)
{
    const auto bytes = M::Bytes(2, 2, 2);
    void* memory = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(memory, MAP_FAILED);
    ASSERT_TRUE(M::Initialize(memory, bytes, 2, 2, 2));
    int ready[2]{};
    ASSERT_EQ(::pipe(ready), 0);
    const auto child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        ::close(ready[0]);
        if (pthread_mutex_lock(TailMetadataTestAccess::Mutex(memory)) != 0) { ::_exit(2); }
        const char marker = 'x';
        if (::write(ready[1], &marker, 1) != 1) { ::_exit(3); }
        ::_exit(0);  // Deliberately die while holding the process-shared mutex.
    }
    ::close(ready[1]);
    pollfd pfd{ready[0], POLLIN, 0};
    const bool readable = ::poll(&pfd, 1, 5000) > 0;
    char marker{};
    const bool signalled = readable && ::read(ready[0], &marker, 1) == 1;
    ::close(ready[0]);
    int status = 0;
    bool exited = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (::waitpid(child, &status, WNOHANG) == child) { exited = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!exited) { ::kill(child, SIGKILL); }
    EXPECT_TRUE(signalled);
    EXPECT_TRUE(exited);
    if (signalled && exited) {
        M metadata(memory);
        M::Ref ref;
        EXPECT_EQ(metadata.Register(Id(1), {}, true, ref), M::Result::Poisoned);
        EXPECT_EQ(metadata.Register(Id(1), {}, true, ref), M::Result::Poisoned);
        EXPECT_EQ(metadata.GetStats().status, M::Result::Poisoned);
    }
    EXPECT_EQ(::munmap(memory, bytes), 0);
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
