/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef GSA_HBM_MANAGER_ADAPT_H
#define GSA_HBM_MANAGER_ADAPT_H

#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <kvcache_log.h>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef DEBUG
#define HBM_ASSERT(f) assert(f)
#else
#define HBM_ASSERT(f) ((void)0)
#endif

// ================= 可配置项  ===============

enum class HbmEvictMode {
    Lru = 0,
    Decay = 1,
};

enum class HbmWindowPlan {
    Uniform = 0,
    Budget4096 = 1,
    Budget5120 = 2,
};

namespace gsa_layer_window {

inline int getWindowSize4096(int layerId) 
{
    if (layerId < 0) { return 5120; }
    if (layerId < 21) { return 3072; }
    if (layerId < 33) { return 4096; }
    if (layerId < 65) { return 5120; }
    return 3072;
}

inline int getWindowSize5120(int layerId) 
{
    if (layerId < 0) { return 5120; }
    if (layerId < 21) { return 4096; }
    if (layerId < 33) { return 5120; }
    if (layerId < 65) { return 5120; }
    return 4096;
}

inline int getWindowSize(HbmWindowPlan plan, int layerId, int immutableSize)
{
    switch (plan) {
        case HbmWindowPlan::Budget4096: return getWindowSize4096(layerId);
        case HbmWindowPlan::Budget5120: return getWindowSize5120(layerId);
        case HbmWindowPlan::Uniform:
        default: return immutableSize;
    }
}

inline HbmEvictMode parseEvictMode(const char* s, HbmEvictMode fallback)
{
    if (s == nullptr || *s == '\0') { return fallback; }
    std::string v(s);
    for (char&c :v) { c = (char)std::tolower((unsigned char)c); }
    if (v == "decay" || v == "1" || v == "layered") { return HbmEvictMode::Decay; }
    if (v == "lru" || v == "0" || v == "original") { return HbmEvictMode::Lru; }
    return fallback;
}

inline HbmWindowPlan parseWindowPlan(const char* s, HbmWindowPlan fallback)
{
    if (s == nullptr || *s == '\0') { return fallback; }
    std::string v(s);
    for (char&c :v) { c = (char)std::tolower((unsigned char)c); }
    if (v == "4096" || v == "budget4096") { return HbmWindowPlan::Budget4096; }
    if (v == "lru" || v == "budget5120") { return HbmWindowPlan::Budget5120; }
    if (v == "uniform" || v == "none" || v == "0") { return HbmWindowPlan::Uniform; }
    return fallback;
}

struct GlobalConfig {
    HbmEvictMode evictMode = HbmEvictMode::Lru;
    HbmWindowPlan windowPlan = HbmWindowPlan::Uniform;
    float decayLambda = 0.85;
    int windowSteps = 20;
    bool loadedFromEnv = false;
};

inline GlobalConfig& globalConfig()
{
    static GlobalConfig cfg;
    return cfg;
}

inline bool resolveConfigFromEnv()
{
    GlobalConfig& cfg = globalConfig();
    if (cfg.loadedFromEnv) { return true; }
    cfg.loadedFromEnv = true;

    cfg.evictMode = HbmEvictMode::Lru;
    cfg.windowPlan = HbmWindowPlan::Uniform;
    cfg.decayLambda = 0.85f;
    cfg.windowSteps = 20;

    bool any = false;
    if (const char* v = std::getenv("UCM_HBM_EVICT_MODE")) {
        cfg.evictMode = parseEvictMode(v, cfg.evictMode);
        any = true;
    }
    if (const char* v = std::getenv("UCM_HBM_WINDOW_PLAN")) {
        cfg.windowPlan = parseWindowPlan(v, cfg.windowPlan);
        any = true;
    }
    if (const char* v = std::getenv("UCM_HBM_DECAY_LAMBDA")) {
        float f = (float)std::atof(v);
        if (f > 0.0f && f <= 1.0f) { cfg.decayLambda = f; any = true; }
    }
    if (const char* v = std::getenv("UCM_HBM_WINDOW_STEPS")) {
        int n = std::atoi(v);
        if (n > 0) { cfg.windowSteps = n; any = true; }
    }
    return any;
}

} // namespace gsa_layer_window

// ================= HBM 管理器 =================
class HBMManagerLayered {
public:
    static constexpr int kAutoEvict = -1;
    static constexpr int kAutoPlan = -1;

    HBMManagerLayered() : mInitialized(false) {}

    void init(int dramTensorCount, int hbmCapacity, int hbmImmutableSize, int topkSize,
              Logger& mLogger, bool preloadHbm = true, int maxTotalSlots = 32768,
            int layerId = -1, int evictMode = kAutoEvict, int windowPlan = kAutoPlan)
    {
        HBM_ASSERT(dramTensorCount > 0);
        HBM_ASSERT(hbmCapacity > 0);
        HBM_ASSERT(hbmImmutableSize > 0);
        HBM_ASSERT(topkSize > 0);
        HBM_ASSERT(hbmImmutableSize <= hbmCapacity);
        HBM_ASSERT(topkSize <= hbmCapacity);

        mDramTensorCount = dramTensorCount;
        mHbmCapacity = hbmCapacity;
        mHbmImmutableSize = hbmImmutableSize;
        mTopkSize = topkSize;
        mMaxTotalSlots = (maxTotalSlots > hbmCapacity) ? maxTotalSlots : hbmCapacity;
        mPinnedSlotCount = 0;
        mHitRate = 0.0f;

        gsa_layer_window::resolveConfigFromEnv();
        const gsa_layer_window::GlobalConfig& gcfg = gsa_layer_window::globalConfig();

        mEvictMode = (evictMode == kAutoEvict)
                        ? gcfg.evictMode
                        : ((evictMode == 1) ? HbmEvictMode::Decay : HbmEvictMode::Lru);
        
        mWindowPlan = (windowPlan == kAutoPlan)
                        ? gcfg.windowPlan
                        : ((windowPlan == 1) ? HbmWindowPlan::Budget4096
                                             : ((windowPlan == 2) ? HbmWindowPlan::Budget5120
                                                                  : HbmWindowPlan::Uniform));
        mDecayLambda = gcfg.decayLambda;
        mWindowSteps = (gcfg.windowSteps >= 0) ? gcfg.windowSteps : 20;

        mLayerId = layerId;
        int win = gsa_layer_window::getWindowSize(mWindowPlan, layerId, hbmImmutableSize);
        if (win < mTopkSize) { win = mTopkSize; }
        if (win > mHbmImmutableSize) { win == mHbmImmutableSize; }
        mWindowSize = win;
        
        mLogger.log(
            LogLevel::DEBUG,
            "|HBM Manager| Init layer : %d, evictMode: %s, windowPlan: %s, "
            "mDramTensorCount: %u, mHbmCapacity: %d, mHbmImmutableSize: %d, mWindowSize: %d"
            "mTopkSize: %d, mMaxTotalSlots: %d lambda %.2f, m: %d\n",
            mLayerId, (mEvictMode == HbmEvictMode::Decay) ? "decay" : "lru",
            (mWindowPlan == HbmWindowPlan::Budget4096) ? "4096"
            : (mWindowPlan == HbmWindowPlan::Budget5120) ? "5120"
                                                         : "uniform",       
            mDramTensorCount, mHbmCapacity, mHbmImmutableSize, mWindowSize, mTopkSize,
            mMaxTotalSlots, mDecayLambda, mWindowSteps);

        mGlobalToSlot.assign(mDramTensorCount, -1);
        mSlotToGlobal.assign(mHbmCapacity, -1);
        mLruPrev.assign(mMaxTotalSlots, -1);
        mLruNext.assign(mMaxTotalSlots, -1);
        mIsPinned.assign(mMaxTotalSlots, false);
        mAddedSlotFlag.assign(mMaxTotalSlots, 0);

        if (mEvictMode == HbmEvictMode::Decay) {
            mDecayScore.assign(mMaxTotalSlots, 0.0f);
            mDecayLastStep.assign(mMaxTotalSlots, -1);
            mActiveFlag.assign(mMaxTotalSlots, false);
            mActiveSlots.reserve(mWindowSteps * mTopkSize);
            rebuildPowTable();
        } else {
            mDecayScore.clear();
            mDecayLastStep.clear();
            mActiveFlag.clear();
            mActiveSlots.clear();
        }

        if (preloadHbm) { preloadSlotMapping(); }
        initLruChain();
        mAddedSlotGeneration = 0;
        mInitialized = true;
    }

    /* 预载映射：slot [0, min(256, mDramTensorCount-1)] 恒等映射 global，
     * 剩余 slot 倒序回填 [mDramTensorCount-1 ...] 倒序的 gid，止步于恒等区间
     *（越界回填会产生幽灵 slot：validateMapping 必败、LRU 误驱逐、双槽
     * gather）。剩余 slot 保持 -1（冷填按需启用）。 */
    void preloadSlotMapping()
    {
        int slotIdx = 0;
        for (; slotIdx <= 256 && slotIdx < mHbmCapacity && slotIdx < mDramTensorCount;
             ++slotIdx) {
            mGlobalToSlot[slotIdx] = slotIdx;
            mSlotToGlobal[slotIdx] = slotIdx;
        }
        const int identityCount = slotIdx;
        for (int gid = mDramTensorCount - 1;
             slotIdx < mHbmCapacity && gid >= identityCount; --gid, ++slotIdx) {
            mGlobalToSlot[gid] = slotIdx;
            mSlotToGlobal[slotIdx] = gid;
        }
    }

    // LRU 链表仅串常规 slots (0 <-> ... <-> mHbmCapacity-1)；pinned 按需动态加入。
    void initLruChain()
    {
        mLruHead = 0;
        mLruTail = mHbmCapacity - 1;
        for (int i = 0; i < mHbmCapacity; ++i) {
            mLruPrev[i] = (i == 0) ? -1 : i - 1;
            mLruNext[i] = (i == mHbmCapacity - 1) ? -1 : i + 1;
        }
    }

  /* topk 查找主路径：代际标记数组去重（覆盖常规/pinned 两段 gid 域）+
     * hit/miss 分类合流单循环 + miss 换入 + 输出组装。dedupedCount 即去重后
     * 序号（负 gid 占序号不出槽）；miss 的 gid 与序号平行入栈供回填免查表；
     * scratch 复用类成员（per-(req,layer) 独占对象，无跨线程共享）。 */
    void processTopkWithDebug(const std::vector<int>& topkIds, std::vector<int>& output,
                              std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                              Logger& mLogger)
    {
        HBM_ASSERT(mInitialized);
        mTopkSize = topkIds.size();
        std::vector<int> topkSlotIds(mTopkSize - 1);
        missGlobalIds.reserve(mTopkSize);
        missSlots.reserve(mTopkSize);
        mAddedSlotGeneration = (mAddedSlotGeneration == INT_MAX) ? 1 : mAddedSlotGeneration + 1;

        std::unordered_set<int> seenGids;
        std::vector<int>  dedupedTopkIds;
        dedupedTopkIds.reserve(mTopkSize);
        for (int i = 0; i < mTopkSize; ++i) {
            int gid = topkIds[i];
            if (seenGids.count(gid)) {
                mLogger.log(LogLevel::ERROR,
                            "|HBM Manager| Duplicate gid in topkIds: %d at index: %d, skipping\n",
                            gid, i);
                continue;
            }
            seenGids.insert(gid);
            dedupedTopkIds.push_back(gid);
        }
        int dedupedSize = dedupedTopkIds.size();

        std::vector<int> missIndices;
        missIndices.reserve(dedupedSize);
        collectTopkHitAndMisses(dedupedTopkIds, topkSlotIds, missGlobalIds, missIndices, mLogger);

        processTopkMisses(dedupedTopkIds, topkSlotIds, missGlobalIds, missSlots, missIndices, mLogger);

        if (mEvictMode == HbmEvictMode::Decay) {
            for (int slot : missSlots) { touchDecay(slot); }
            pruneDecay();
            buildTopkOutputDecay(topkSlotIds, dedupedSize, output);
        } else {
            buildTopkOutputLru(topkSlotIds, dedupedSize, output);
        }
    }

    int getSlotForGlobal(int gid) const
    {
        if (gid >= mDramTensorCount) { return gid - mDramTensorCount + mHbmCapacity; }
        if (gid >= 0 && gid < mDramTensorCount) { return mGlobalToSlot[gid]; }
        return -1;
    }

    // pinned slot -> gid: 由公式推导，gid = slot - mHbmCapacity + mDramTensorCount
    int getGidForSlot(int slot) const
    {
        if (slot >= 0 && slot < mHbmCapacity) { return mSlotToGlobal[slot]; }
        if (slot >= mHbmCapacity && slot < mMaxTotalSlots && mIsPinned[slot]) {
            return slot - mHbmCapacity + mDramTensorCount;
        }
        return -1;
    }

    bool isInitialized() const { return mInitialized; }
    int getTopkSize() const { return mTopkSize; }
    int getHbmCapacity() const { return mHbmCapacity; }
    int getHbmImmutableSize() const { return mHbmImmutableSize; }
    int getWindowSize() const { return mWindowSize; }
    int getLayerId() const { return mLayerId; }
    int getDramTensorCount() const { return mDramTensorCount; }
    int getPinnedSlotCount() const { return mPinnedSlotCount; }
    int getMaxTotalSlots() const { return mMaxTotalSlots; }
    int getActiveSlotCount() const { return (int)mActiveSlots.size(); }
    bool idDecayMode() const { return mEvictMode == HbmEvictMode::Decay; }

    const char* getEvictModeName() const
    {
        return (mEvictMode == HbmEvictMode::Decay) ? "decay" : "lru";
    }

    const char* getWindowPlanName() const
    {
        switch (mWindowPlan)
        {
            case HbmWindowPlan::Budget4096: return "4096";
            case HbmWindowPlan::Budget5120: return "5120";
            default: return "uniform";
        }
    }

    void setEvictMode(HbmEvictMode mode) { mEvictMode = mode; }

    void setWindowSize(int win) {
        if (win < mTopkSize) { win = mTopkSize; }
        if (win > mHbmImmutableSize) { win = mHbmImmutableSize; }
        mWindowSize = win;
    }

    void setDecayLambda(float lam) { mDecayLambda = lam; }
    void setWindowSteps(int steps) {
        mWindowSteps = (steps > 0) ? steps : 20;
        rebuildPowTable();
    }
    bool isNoCapMode() const { return mWindowSteps == 0; }

    bool validateLruChain() const
    {
        int expectedCount = mHbmCapacity + mPinnedSlotCount;
        int curr = mLruHead;
        int count = 0;
        int safetyLimit = mMaxTotalSlots + 10;

        while (curr != -1 && count < safetyLimit) {
            count++;
            curr = mLruNext[curr];
        }

        if (count >= safetyLimit || count != expectedCount) return false;

        curr = mLruTail;
        count = 0;
        while (curr != -1 && count < safetyLimit) {
            count++;
            curr = mLruPrev[curr];
        }

        return count == expectedCount;
    }

    bool validateMapping() const
    {
        // 检查常规 slot 映射
        for (int slot = 0; slot < mHbmCapacity; ++slot) {
            int gid = mSlotToGlobal[slot];
            if (gid == -1) continue;
            if (gid < -1 || gid >= mDramTensorCount) return false;
            if (mGlobalToSlot[gid] != slot) return false;
        }
        // 检查已初始化的 pinned slot 映射 (gid 由公式推导)
        for (int slot = mHbmCapacity; slot < mMaxTotalSlots; ++slot) {
            if (!mIsPinned[slot]) continue;
            int expectedGid = slot - mHbmCapacity + mDramTensorCount;
            // pinned slot 的 gid 必须与公式一致
            if (expectedGid < mDramTensorCount) return false;
        }
        return true;
    }

    // mSlotToGlobal 仅覆盖常规 slot [0, mHbmCapacity-1]
    // LastChunkMove 等外部代码仅遍历此范围
    std::vector<int> mSlotToGlobal;
    std::vector<int> mGlobalToSlot;
    float mHitRate;

private:
    int mDramTensorCount;
    int mHbmCapacity;
    int mHbmImmutableSize;
    int mTopkSize;
    int mPinnedSlotCount;
    int mMaxTotalSlots;
    int mLayerId;
    int mWindowSize;
    HbmEvictMode mEvictMode = HbmEvictMode::Lru;
    HbmWindowPlan mWindowPlan = HbmWindowPlan::Uniform;

    std::vector<int> mLruPrev;
    std::vector<int> mLruNext;
    std::vector<bool> mIsPinned;
    std::vector<int> mAddedSlotFlag;

    int mLruHead;
    int mLruTail;
    int mAddedSlotGeneration;
    bool mInitialized;

    std::vector<float> mDecayScore;
    std::vector<int> mDecayLastStep;
    std::vector<bool> mActiveFlag;
    std::vector<int> mActiveSlots;
    int mDecayStep = 0;
    float mDecayLambda = 0.85f;
    int mWindowSteps = 20;

    static constexpr float kMinDecayScore = 1e-3f;
    static constexpr int kPowTableSize = 512;
    std::vector<float> mPowTable;
    std::vector<std::pair<float, int>> mRankedBuf;

    inline bool powLambda(int k) const
    {
        if (k > 0 && k < (int)mPowTable.size()) { return mPowTable[k]; }
        return std::pow(mDecayLambda, (float)k);
    }

    inline bool isDecayAlive(int slot, int age) const
    {
        if (age < 0) { return false; }
        if (mWindowSteps > 0) { return age < mWindowSteps; }
        return mDecayScore[slot] * powLambda(age) >= kMinDecayScore;
    }

    void rebuildPowTable()
    {
        int n = (mWindowSteps > 0) ? std::max(mWindowSteps, 64) : kPowTableSize;
        mPowTable.resize(n + 1);
        mPowTable[0] = 1.0f;
        for (int k = 0; k < n; ++k) {
            mPowTable[k] = mPowTable[k - 1] * mDecayLambda;
        }
    }

    /* miss 换入与回填：victim 自 LRU 尾部驱逐（跳过 pinned）。missGlobalIds 与
     * missIndices 平行入栈（missIndices[j] 为第 j 个 miss 的去重后序号），回填
     * 直接取 missGlobalIds[j]，无需 deduped 序列二次查表。 */
    void collectTopkHitAndMisses(std::vector<int>& dedupedTopkIds,
                                 std::vector<int>& topkSlotIds, std::vector<int>& missGlobalIds,
                                 std::vector<int>& missIndices, Logger& mLogger)
    {
        int dedupedSize = dedupedTopkIds.size();
        for (int i = 0; i < dedupedSize; ++i) {
            int gid = dedupedTopkIds[i];
            if (gid > mDramTensorCount) {
                int transformedSlot = gid - mDramTensorCount + mHbmCapacity;
                if (transformedSlot >= mMaxTotalSlots) {
                    mLogger.log(LogLevel::ERROR,
                                "|HBM Manager| Pinned slot %d exceeds maxTotalSlots %d, gid: %d\n",
                                transformedSlot, mMaxTotalSlots, gid);
                    topkSlotIds[i] = transformedSlot;
                    continue;
                }
                if (!mIsPinned[transformedSlot]) {
                    mIsPinned[transformedSlot] = true;
                    insertToHead(transformedSlot);
                    mPinnedSlotCount++;
                } else {
                    moveToHead(transformedSlot);
                }
                topkSlotIds[i] = transformedSlot;
                mAddedSlotFlag[transformedSlot] = mAddedSlotGeneration;
                touchDecay(transformedSlot);
                continue;
            }
            if (gid < 0) {
                mLogger.log(LogLevel::ERROR, "|KVCache Prefetch| Topk is abnormal value: %d \n", gid);
            }
            int slot = mGlobalToSlot[gid];
            int threshold = 2048;
            if (slot != -1) {
                moveToHead(slot);
                topkSlotIds[i] = slot;
                mAddedSlotFlag[slot] = mAddedSlotGeneration;
                touchDecay(slot);
            } else {
                missGlobalIds.push_back(gid);
                missIndices.push_back(i);
                if ((int)missIndices.size() == threshold) { break; }
            }
            int thresholdSub = 2047;
            if (i == thresholdSub) {
                mHitRate = 1 - missIndices.size() * 1.0 / threshold;
            }
        }
    }

    void processTopkMisses(const std::vector<int>& dedupedTopkIds, std::vector<int>& topkSlotIds,
                           std::vector<int>& missGlobalIds, std::vector<int>& missSlots,
                           const std::vector<int>& missIndices, Logger& mLogger)
    {
        int missCount = missGlobalIds.size();
        int loadedCount = 0;
        if (missCount > 0) {
            for (int i = 0; i < missCount; ++i) {
                int victimSlot = mLruTail;
                while (victimSlot != -1 && mIsPinned[victimSlot]) {
                    victimSlot = mLruPrev[victimSlot];
                }
                if (victimSlot == -1) {
                    mLogger.log(LogLevel::ERROR,
                                "|HBM Manager| No evictable slot for miss gid: %d, "
                                "remaining %d misses dropped\n",
                                missGlobalIds[i], missCount - i);
                    continue;
                }
                int oldGid = mSlotToGlobal[victimSlot];
                if (oldGid >= 0 && oldGid < mDramTensorCount) {
                    mGlobalToSlot[oldGid] = -1;
                } else if (oldGid == -1) {
                    mLogger.log(LogLevel::WARNING, "|HBM Manager| Evicting unallocated slot: %d\n",
                                victimSlot);
                } else {
                    mLogger.log(
                        LogLevel::ERROR,
                        "|HBM Manager| Evicting slot with out-of-range gid: %d in slot: %d\n",
                        oldGid, victimSlot);
                }
                clearDecay(victimSlot);

                int newGid = missGlobalIds[i];
                mSlotToGlobal[victimSlot] = newGid;
                mGlobalToSlot[newGid] = victimSlot;
                missSlots.push_back(victimSlot);
                loadedCount++;
                removeFromList(victimSlot);
                insertToHead(victimSlot);
            }
            missGlobalIds.resize(loadedCount);
            for (int j = 0; j < loadedCount; ++j) {
                int idx = missIndices[j];
                int gid = dedupedTopkIds[idx];
                int slot = mGlobalToSlot[gid];
                topkSlotIds[idx] = slot;
                if (slot >= 0 && slot < mMaxTotalSlots) {
                    mAddedSlotFlag[slot] = mAddedSlotGeneration;
                }
            }
        }
    }

    void buildTopkOutputLru(const std::vector<int>& topkSlotIds, int dedupedSize,
                         std::vector<int>& output)
    {
        output.reserve(mHbmImmutableSize);
        int realTopkSlotNum = 0;
        for (int i = 0; i < dedupedSize; ++i) {
            if (topkSlotIds[i] < 0) { continue; }
            output.push_back(topkSlotIds[i]);
            realTopkSlotNum += 1;
        }
        int slotsNeeded = mHbmImmutableSize - realTopkSlotNum;
        int count = 0;
        int curr = mLruHead;
        while (curr != -1 && count < slotsNeeded) {
            if (mAddedSlotFlag[curr] != mAddedSlotGeneration) {
                output.push_back(curr);
                mAddedSlotFlag[curr] = mAddedSlotGeneration;
                count++;
            }
            curr = mLruNext[curr];
        }
    }

    void buildTopkOutputDecay(const std::vector<int>& topkSlotIds, int dedupedSize,
                              std::vector<int>& output)
    {
        output.clear();
        output.reserve(mHbmImmutableSize);

        int realTopkSlotNum = 0;
        for (int i = 0; i < dedupedSize; ++i) {
            int slot = topkSlotIds[i];
            if (slot < 0) { continue; }
            output.push_back(topkSlotIds[i]);
            mAddedSlotFlag[slot] = mAddedSlotGeneration;
            realTopkSlotNum += 1;
        }

        int histNeeded = mHbmImmutableSize - realTopkSlotNum;
        if (histNeeded > 0 && !mActiveSlots.empty()) {
            mRankedBuf.clear();
            for (int slot : mActiveSlots) {
                if (mAddedSlotFlag[slot] == mAddedSlotGeneration) { continue; }
                int age = mDecayStep - mDecayLastStep[slot];
                if (!isDecayAlive(slot, age)) { continue; }
                float eff = mDecayScore[slot] * powLambda(age);
                if (eff <= 0.0f) { continue; }
                mRankedBuf.emplace_back(eff, slot);
            }
        }
        auto& ranked = mRankedBuf;
        if ((int)ranked.size() > histNeeded) {
            std::nth_element(ranked.begin(), ranked.begin() + histNeeded, ranked.end(),
                            [](const std::pair<float, int>& a,
                            const std::pair<float, int>& b) {
                                if (a.first != b.first) return a.first > b.first;
                                return a.second < b.second; 
                            });
            ranked.resize(histNeeded);
        }
        std::sort(ranked.begin(), ranked.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second; 
                });
        
        for (const auto& kv : ranked) {
            if ((int)output.size() >= mWindowSize) { break; }
            if (mAddedSlotFlag[kv.second] == mAddedSlotGeneration) { continue; }
            output.push_back(kv.second);
            mAddedSlotFlag[kv.second] = mAddedSlotGeneration;
        }

        while ((int)output.size() < mHbmImmutableSize) {
            output.push_back(-1);
        }

        if ((int)output.size() > mHbmImmutableSize) {
            output.resize(mHbmImmutableSize);
        }
    }

    inline void touchDecay(int slot)
    {
        if (mEvictMode != HbmEvictMode::Decay) { return; }
        if (slot < 0 || slot >= mMaxTotalSlots) { return; }
        if (mDecayLastStep[slot] >= 0) {
            int age = mDecayStep - mDecayLastStep[slot];
            if (age > 0) {
                mDecayScore[slot] *= powLambda(age);
            }
        }
        mDecayScore[slot] += 1.0f;
        mDecayLastStep[slot] = mDecayStep;
        if (!mActiveFlag[slot]) {
            mActiveFlag[slot] = true;
            mActiveSlots.push_back(slot);
        }
    }

    inline void clearDecay(int slot)
    {
        if (mEvictMode == HbmEvictMode::Decay) { return; }
        if (slot < 0 || slot >= mMaxTotalSlots) { return; }
        mDecayScore[slot] = 0.0f;
        mDecayLastStep[slot] = -1;
    }

    void pruneDecay()
    {
        if (mEvictMode == HbmEvictMode::Decay) { return; }
        int write = 0;
        for (int read = 0; read < (int)mActiveSlots.size(); ++ read) {
            int slot = mActiveSlots[read];
            int last = mDecayLastStep[slot];
            int age = mDecayStep - last;
            if (last < 0 || !isDecayAlive(slot, age)) {
                mActiveFlag[slot] = false;
                mDecayScore[slot] = 0.0f;
                mDecayLastStep[slot] = -1;
                continue;
            }
            mActiveSlots[write++] = slot;
        }
        mActiveSlots.resize(write);
    }

    inline void removeFromList(int slot)
    {
        int p = mLruPrev[slot];
        int n = mLruNext[slot];
        if (p != -1)
            mLruNext[p] = n;
        else
            mLruHead = n;
        if (n != -1)
            mLruPrev[n] = p;
        else
            mLruTail = p;
    }

    inline void insertToHead(int slot)
    {
        mLruPrev[slot] = -1;
        mLruNext[slot] = mLruHead;
        if (mLruHead != -1) mLruPrev[mLruHead] = slot;
        mLruHead = slot;
        if (mLruTail == -1) mLruTail = slot;
    }

    inline void moveToHead(int slot)
    {
        if (slot == mLruHead) return;
        removeFromList(slot);
        insertToHead(slot);
    }
};

#endif  // GSA_HBM_MANAGER_H