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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_PREFIX_TAIL_EVICTION_POLICY_H
#define UNIFIEDCACHE_DRAM_STORE_CC_PREFIX_TAIL_EVICTION_POLICY_H

#include <algorithm>
#include <chrono>
#include <vector>
#include "entry.h"
#include "eviction_policy.h"
#include "logger/logger.h"

namespace UC::DramPool {

/**
 * @brief Orders entries for tail-first prefix eviction.
 *
 * Entry::position carries the block's prefix-chain depth (prefixDepth
 * semantics). Entries are ordered by depth descending so the tail of every
 * chain is evicted first, tiebroken by lifeTimeout ascending. Unmarked
 * entries (Detail::kUnmarkedPrefixDepth) sort as the deepest candidates
 * and are therefore evicted first.
 */
struct PrefixTailEntryCmp {
    bool operator()(const EntryPtr& a, const EntryPtr& b) const noexcept
    {
        if (a->position != b->position) { return a->position > b->position; }
        return a->lifeTimeout < b->lifeTimeout;
    }
};

/**
 * @brief Eviction policy that reclaims the deepest tail blocks first.
 *
 * Chain hashes make stored blocks a prefix trie: a block's depth equals its
 * position in every chain that contains it. Ordering victims by depth
 * descending evicts each request's tail first while shared shallow prefixes
 * (the head every future hit depends on) survive longest. GetEvictionResults
 * evicts up to evict_ratio * size() entries from the front of the ordering,
 * skipping protected entries and continuing with shallower candidates until
 * the target is reached.
 */
class PrefixTailEvictionPolicy : public OrderedEvictionPolicy<PrefixTailEntryCmp> {
public:
    std::vector<EntryPtr> GetEvictionResults(double evict_ratio) override
    {
        std::vector<EntryPtr> victims;
        if (evict_ratio == 0.0 || entries_.empty()) { return victims; }

        const auto now = std::chrono::system_clock::now();
        const auto target = std::max<std::size_t>(
            1, static_cast<std::size_t>(static_cast<double>(entries_.size()) * evict_ratio));
        for (const auto& entry : entries_) {
            if (victims.size() >= target) { break; }
            if (!entry->TryMarkEvicting(now)) { continue; }
            victims.push_back(entry);
        }
        if (!victims.empty()) {
            UC_INFO("PrefixTailEvictionPolicy evict {} of {} entries.", victims.size(),
                    entries_.size());
        }
        return victims;
    }
};

}  // namespace UC::DramPool

#endif
