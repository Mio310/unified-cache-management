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
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "dram/cc/drampool/entry.h"
#include "dram/cc/drampool/prefix_tail_eviction_policy.h"
#include "dram/dram_test_common.h"

using UC::Status;
using UC::DramPool::PrefixTailEvictionPolicy;
using UC::Test::Dram::Clock;
using UC::Test::Dram::EntryStatus;
using UC::Test::Dram::KeyFromHex;
using UC::Test::Dram::MakeEntry;
using UC::Test::Dram::TimePoint;

class UCPrefixTailEvictionPolicyTest : public testing::Test {
protected:
    PrefixTailEvictionPolicy policy_;
    const TimePoint past_ = Clock::now() - std::chrono::seconds(10);
    const TimePoint future_ = Clock::now() + std::chrono::seconds(10);

    std::vector<std::uint32_t> VictimPositions(const std::vector<UC::DramPool::EntryPtr>& victims)
    {
        std::vector<std::uint32_t> positions;
        positions.reserve(victims.size());
        for (const auto& victim : victims) { positions.push_back(victim->position); }
        return positions;
    }
};

TEST_F(UCPrefixTailEvictionPolicyTest, AddKeyReturnsOk)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
}

TEST_F(UCPrefixTailEvictionPolicyTest, AddKeyDuplicateReturnsDuplicateKey)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_EQ(policy_.AddKey(key, MakeEntry(key, 0, past_)), Status::DuplicateKey());
}

TEST_F(UCPrefixTailEvictionPolicyTest, AddKeyNullptrReturnsInvalidParam)
{
    auto key = KeyFromHex("a1");
    ASSERT_EQ(policy_.AddKey(key, nullptr), Status::InvalidParam());
}

TEST_F(UCPrefixTailEvictionPolicyTest, DeleteKeyReturnsOkAndSecondIsNotFound)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_TRUE(policy_.DeleteKey(key).Success());
    ASSERT_EQ(policy_.DeleteKey(key), Status::NotFound());
}

TEST_F(UCPrefixTailEvictionPolicyTest, AccessKeyReturnsOk)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 0, past_)).Success());
    ASSERT_TRUE(policy_.AccessKey(key).Success());
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsEmptyWhenNoEntries)
{
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsEmptyWhenEvictRatioIsZero)
{
    auto key = KeyFromHex("a1");
    ASSERT_TRUE(policy_.AddKey(key, MakeEntry(key, 1, past_)).Success());

    EXPECT_TRUE(policy_.GetEvictionResults(0.0).empty());
}

TEST_F(UCPrefixTailEvictionPolicyTest, SmallPositiveRatioEvictsAtLeastOneEntry)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2, 2, past_)).Success());

    auto victims = policy_.GetEvictionResults(0.1);

    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0]->key, k2);
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsRespectsEvictRatio)
{
    auto k1 = KeyFromHex("a1");
    auto k2 = KeyFromHex("a2");
    auto k3 = KeyFromHex("a3");
    auto k4 = KeyFromHex("a4");
    ASSERT_TRUE(policy_.AddKey(k1, MakeEntry(k1, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k2, MakeEntry(k2, 2, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k3, MakeEntry(k3, 3, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(k4, MakeEntry(k4, 4, past_)).Success());
    auto victims = policy_.GetEvictionResults(0.5);
    ASSERT_EQ(victims.size(), 2UL);
    EXPECT_EQ(victims[0]->key, k4);
    EXPECT_EQ(victims[1]->key, k3);
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsOrdersByPrefixDepthDescending)
{
    // Three chains interleaved on registration: A(1,2,3), B(1,2), C(1).
    auto a1 = KeyFromHex("b1");
    auto b1 = KeyFromHex("b2");
    auto c1 = KeyFromHex("b3");
    auto a2 = KeyFromHex("b4");
    auto b2 = KeyFromHex("b5");
    auto a3 = KeyFromHex("b6");
    ASSERT_TRUE(policy_.AddKey(a1, MakeEntry(a1, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(b1, MakeEntry(b1, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(c1, MakeEntry(c1, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(a2, MakeEntry(a2, 2, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(b2, MakeEntry(b2, 2, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(a3, MakeEntry(a3, 3, past_)).Success());

    auto victims = policy_.GetEvictionResults(1.0);

    ASSERT_EQ(victims.size(), 6UL);
    EXPECT_EQ(VictimPositions(victims), (std::vector<std::uint32_t>{3, 2, 2, 1, 1, 1}));
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsKeepsSharedPrefixHeadLast)
{
    // A and B share the depth-0 head block (same key, registered once).
    auto sharedHead = KeyFromHex("c1");
    auto a1 = KeyFromHex("c2");
    auto a2 = KeyFromHex("c3");
    auto b1 = KeyFromHex("c4");
    ASSERT_TRUE(policy_.AddKey(sharedHead, MakeEntry(sharedHead, 0, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(a1, MakeEntry(a1, 1, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(a2, MakeEntry(a2, 2, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(b1, MakeEntry(b1, 1, past_)).Success());

    auto victims = policy_.GetEvictionResults(0.75);

    ASSERT_EQ(victims.size(), 3UL);
    for (const auto& victim : victims) { EXPECT_NE(victim->key, sharedHead); }
    EXPECT_EQ(VictimPositions(victims), (std::vector<std::uint32_t>{2, 1, 1}));
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsTiebreaksByLifeTimeoutAscending)
{
    auto kLater = KeyFromHex("a1");
    auto kEarlier = KeyFromHex("a2");
    ASSERT_TRUE(policy_.AddKey(kLater, MakeEntry(kLater, 5, future_)).Success());
    ASSERT_TRUE(policy_.AddKey(kEarlier, MakeEntry(kEarlier, 5, past_)).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 2UL);
    EXPECT_EQ(victims[0]->key, kEarlier);
    EXPECT_EQ(victims[1]->key, kLater);
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsEvictsUnmarkedDepthFirst)
{
    constexpr std::uint32_t kUnmarked = UC::Detail::kUnmarkedPrefixDepth;
    auto kUnmarkedKey = KeyFromHex("a1");
    auto kDeepKey = KeyFromHex("a2");
    ASSERT_TRUE(policy_.AddKey(kUnmarkedKey, MakeEntry(kUnmarkedKey, kUnmarked, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(kDeepKey, MakeEntry(kDeepKey, 100, past_)).Success());

    auto victims = policy_.GetEvictionResults(0.5);

    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0]->key, kUnmarkedKey);
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsSkipsNonReadyAndContinues)
{
    auto kDeleting = KeyFromHex("a1");
    auto kReady = KeyFromHex("a2");
    ASSERT_TRUE(
        policy_.AddKey(kDeleting, MakeEntry(kDeleting, 5, past_, EntryStatus::DELETING)).Success());
    ASSERT_TRUE(policy_.AddKey(kReady, MakeEntry(kReady, 1, past_, EntryStatus::READY)).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0]->key, kReady);
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsSkipsNonZeroRefCnt)
{
    auto k = KeyFromHex("a1");
    ASSERT_TRUE(
        policy_.AddKey(k, MakeEntry(k, 1, past_, EntryStatus::READY, /*refCnt=*/1)).Success());
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsSkipsLeasedEntry)
{
    auto k = KeyFromHex("a1");
    ASSERT_TRUE(
        policy_.AddKey(k, MakeEntry(k, 1, past_, EntryStatus::READY, 0, future_)).Success());
    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsMarksVictimsAsDeleting)
{
    auto k = KeyFromHex("a1");
    auto entry = MakeEntry(k, 1, past_);
    ASSERT_TRUE(policy_.AddKey(k, entry).Success());
    auto victims = policy_.GetEvictionResults(1.0);
    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(entry->status, EntryStatus::DELETING);
}

TEST_F(UCPrefixTailEvictionPolicyTest, LeasedTailStopsBeforeShallowerEntries)
{
    auto kTail = KeyFromHex("a1");
    auto kSameDepth = KeyFromHex("a2");
    auto kHead = KeyFromHex("a3");
    ASSERT_TRUE(
        policy_.AddKey(kTail, MakeEntry(kTail, 5, past_, EntryStatus::READY, 0, future_)).Success());
    ASSERT_TRUE(policy_.AddKey(kSameDepth, MakeEntry(kSameDepth, 5, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(kHead, MakeEntry(kHead, 1, past_)).Success());

    auto victims = policy_.GetEvictionResults(1.0);

    ASSERT_EQ(victims.size(), 1UL);
    EXPECT_EQ(victims[0]->key, kSameDepth);
}

TEST_F(UCPrefixTailEvictionPolicyTest, LeasedTailAloneDoesNotEvictHead)
{
    auto kTail = KeyFromHex("a1");
    auto kHead = KeyFromHex("a2");
    ASSERT_TRUE(
        policy_.AddKey(kTail, MakeEntry(kTail, 5, past_, EntryStatus::READY, 0, future_)).Success());
    ASSERT_TRUE(policy_.AddKey(kHead, MakeEntry(kHead, 1, past_)).Success());

    EXPECT_TRUE(policy_.GetEvictionResults(1.0).empty());
}

TEST_F(UCPrefixTailEvictionPolicyTest, GetEvictionResultsFillsTargetFromShallowerEntries)
{
    // The two deepest blocks are pinned by refCnt; the third victim must come
    // from a shallower depth, proving the scan continues past skipped entries.
    auto kDeep = KeyFromHex("a1");
    auto kDeep2 = KeyFromHex("a2");
    auto kMid = KeyFromHex("a3");
    auto kShallow = KeyFromHex("a4");
    auto kHead = KeyFromHex("a5");
    ASSERT_TRUE(
        policy_.AddKey(kDeep, MakeEntry(kDeep, 5, past_, EntryStatus::READY, /*refCnt=*/1))
            .Success());
    ASSERT_TRUE(
        policy_.AddKey(kDeep2, MakeEntry(kDeep2, 4, past_, EntryStatus::READY, /*refCnt=*/1))
            .Success());
    ASSERT_TRUE(policy_.AddKey(kMid, MakeEntry(kMid, 3, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(kShallow, MakeEntry(kShallow, 2, past_)).Success());
    ASSERT_TRUE(policy_.AddKey(kHead, MakeEntry(kHead, 1, past_)).Success());

    auto victims = policy_.GetEvictionResults(0.6);

    ASSERT_EQ(victims.size(), 3UL);
    EXPECT_EQ(VictimPositions(victims), (std::vector<std::uint32_t>{3, 2, 1}));
}
