/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include "cache/cc/load_queue.h"
#include "detail/data_generator.h"
#include "detail/mock_store.h"
#include "detail/random.h"
#include "detail/types_helper.h"

class UCCacheLoadQueueTest : public testing::Test {
public:
    UC::Test::Detail::Random rd;
    static UC::Detail::TaskHandle NextId()
    {
        static std::atomic<size_t> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    }
};

TEST_F(UCCacheLoadQueueTest, LoadSameBlockTwice)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke(NextId));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    size_t tensorSize = 32768;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task1 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter1 = std::make_shared<UC::Latch>();
    loadQ.Submit(task1, waiter1);
    waiter1->Wait();
    ASSERT_FALSE(failureSet.Contains(task1->id));
    auto task2 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter2 = std::make_shared<UC::Latch>();
    loadQ.Submit(task2, waiter2);
    waiter2->Wait();
    ASSERT_FALSE(failureSet.Contains(task2->id));
}

TEST_F(UCCacheLoadQueueTest, SharedFailureStopsNonOwnerWait)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).Times(0);
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    size_t tensorSize = 32768;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    auto owner = buffer.Get(blockId, shardIdx, true, true);
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);

    owner.MarkFailed(UC::Status::NotFound());

    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_TRUE(failureSet.Contains(task->id));
    ASSERT_EQ(task->FailureStatus(), UC::Status::NotFound());
}

TEST_F(UCCacheLoadQueueTest, LoadWhileBackendSubmitFailed)
{
    using namespace UC::CacheStore;
    using namespace testing;
    std::promise<void> submitEntered;
    std::promise<void> allowSubmitFailure;
    auto allowSubmitFailureFuture = allowSubmitFailure.get_future().share();
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load)
        .WillOnce(Invoke([&](UC::Detail::TaskDesc) -> UC::Expected<UC::Detail::TaskHandle> {
            submitEntered.set_value();
            allowSubmitFailureFuture.wait();
            return UC::Status::NotFound();
        }));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    size_t tensorSize = 32768;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    submitEntered.get_future().wait();
    auto observer = buffer.Get(blockId, shardIdx, true, true);

    allowSubmitFailure.set_value();

    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_TRUE(failureSet.Contains(task->id));
    ASSERT_EQ(observer.GetState(), TransBuffer::State::FAILED);
    ASSERT_EQ(observer.FailureStatus(), UC::Status::NotFound());
    ASSERT_EQ(task->FailureStatus(), UC::Status::NotFound());
}

TEST_F(UCCacheLoadQueueTest, LoadWhileBackendWaitFailed)
{
    using namespace UC::CacheStore;
    using namespace testing;
    std::promise<void> waitEntered;
    std::promise<void> allowWaitFailure;
    auto allowWaitFailureFuture = allowWaitFailure.get_future().share();
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke(NextId));
    EXPECT_CALL(backend, Wait).WillOnce(Invoke([&](UC::Detail::TaskHandle) {
        waitEntered.set_value();
        allowWaitFailureFuture.wait();
        return UC::Status::NotFound();
    }));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    size_t tensorSize = 32768;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    waitEntered.get_future().wait();
    auto observer = buffer.Get(blockId, shardIdx, true, true);

    allowWaitFailure.set_value();

    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_TRUE(failureSet.Contains(task->id));
    ASSERT_EQ(observer.GetState(), TransBuffer::State::FAILED);
    ASSERT_EQ(observer.FailureStatus(), UC::Status::NotFound());
    ASSERT_EQ(task->FailureStatus(), UC::Status::NotFound());
}

TEST_F(UCCacheLoadQueueTest, DdrLoadCopiesCacheSlotIntoCallerBuffer)
{
    using namespace UC::CacheStore;
    constexpr size_t tensorSize = 4096;
    std::vector<std::byte> payload(tensorSize, std::byte{0x5a});
    std::vector<std::byte> first(tensorSize, std::byte{0});
    std::vector<std::byte> second(tensorSize, std::byte{0});
    void* cacheAddr = nullptr;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
        EXPECT_EQ(task.memoryType, UC::Detail::TaskDesc::MemoryType::DDR);
        EXPECT_EQ(task.size(), 1U);
        EXPECT_EQ(task.front().addrs.size(), 1U);
        cacheAddr = task.front().addrs.front();
        EXPECT_NE(cacheAddr, first.data());
        std::memcpy(cacheAddr, payload.data(), tensorSize);
        return NextId();
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = tensorSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("c1b2c3d4e5f6789012345678901234ab");

    UC::Detail::TaskDesc miss;
    miss.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    miss.push_back(UC::Detail::Shard{blockId, 0, {first.data()}});
    auto missTask = std::make_shared<TransTask>(TransTask::Type::LOAD, std::move(miss));
    auto missWaiter = std::make_shared<UC::Latch>();
    loadQ.Submit(missTask, missWaiter);
    ASSERT_TRUE(missWaiter->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(missTask->id));
    EXPECT_EQ(std::memcmp(first.data(), payload.data(), tensorSize), 0);

    UC::Detail::TaskDesc hit;
    hit.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    hit.push_back(UC::Detail::Shard{blockId, 0, {second.data()}});
    auto hitTask = std::make_shared<TransTask>(TransTask::Type::LOAD, std::move(hit));
    auto hitWaiter = std::make_shared<UC::Latch>();
    loadQ.Submit(hitTask, hitWaiter);
    ASSERT_TRUE(hitWaiter->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(hitTask->id));
    EXPECT_EQ(std::memcmp(second.data(), payload.data(), tensorSize), 0);

    auto handle = buffer.Get(blockId, 0, true, true);
    EXPECT_TRUE(handle.Ready());
    EXPECT_TRUE(handle.Owner());
    EXPECT_EQ(std::memcmp(handle.Data(), payload.data(), tensorSize), 0);
}

TEST_F(UCCacheLoadQueueTest, DdrLoadOnTokenSplicesSelectedRangesIntoCallerBuffer)
{
    using namespace UC::CacheStore;
    constexpr size_t tensorSize = 4096;
    std::vector<std::byte> payload(tensorSize, std::byte{0x5a});
    std::vector<std::byte> first(tensorSize, std::byte{0});
    std::vector<std::byte> second(tensorSize, std::byte{0});
    void* cacheAddr = nullptr;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
        EXPECT_EQ(task.memoryType, UC::Detail::TaskDesc::MemoryType::DDR);
        EXPECT_EQ(task.size(), 1U);
        EXPECT_EQ(task.front().addrs.size(), 1U);
        cacheAddr = task.front().addrs.front();
        EXPECT_NE(cacheAddr, first.data());
        std::memcpy(cacheAddr, payload.data(), tensorSize);
        return NextId();
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = tensorSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("d1b2c3d4e5f6789012345678901234ab");
    std::vector<UC::Detail::TokenFilter> filter(1);
    filter[0].ranges = {{UC::Detail::TokenRange{0, tensorSize / 2, 0}}};

    UC::Detail::TaskDesc miss;
    miss.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    miss.push_back(UC::Detail::Shard{blockId, 0, {first.data()}});
    auto missTask =
        std::make_shared<TransTask>(TransTask::Type::LOAD_ON_TOKEN, std::move(miss), filter);
    auto missWaiter = std::make_shared<UC::Latch>();
    loadQ.Submit(missTask, missWaiter);
    ASSERT_TRUE(missWaiter->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(missTask->id));
    /* The selected token range is spliced into the caller buffer; the rest of
     * the buffer stays untouched. */
    EXPECT_EQ(std::memcmp(first.data(), payload.data(), tensorSize / 2), 0);
    EXPECT_TRUE(std::all_of(first.begin() + tensorSize / 2, first.end(),
                            [](std::byte b) { return b == std::byte{0}; }));

    UC::Detail::TaskDesc hit;
    hit.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    hit.push_back(UC::Detail::Shard{blockId, 0, {second.data()}});
    auto hitTask =
        std::make_shared<TransTask>(TransTask::Type::LOAD_ON_TOKEN, std::move(hit), filter);
    auto hitWaiter = std::make_shared<UC::Latch>();
    loadQ.Submit(hitTask, hitWaiter);
    ASSERT_TRUE(hitWaiter->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(hitTask->id));
    EXPECT_EQ(std::memcmp(second.data(), payload.data(), tensorSize / 2), 0);
    EXPECT_TRUE(std::all_of(second.begin() + tensorSize / 2, second.end(),
                            [](std::byte b) { return b == std::byte{0}; }));

    auto handle = buffer.Get(blockId, 0, true, true);
    EXPECT_TRUE(handle.Ready());
    EXPECT_TRUE(handle.Owner());
    EXPECT_EQ(std::memcmp(handle.Data(), payload.data(), tensorSize), 0);
}

TEST_F(UCCacheLoadQueueTest, DdrLoadOnTokenBackendWaitFailureFailsTask)
{
    using namespace UC::CacheStore;
    constexpr size_t tensorSize = 4096;
    std::vector<std::byte> dst(tensorSize, std::byte{0});
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke(NextId));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::NotFound()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = tensorSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("e1b2c3d4e5f6789012345678901234ab");
    std::vector<UC::Detail::TokenFilter> filter(1);
    filter[0].ranges = {
        {UC::Detail::TokenRange{0, tensorSize, 0}}
    };

    UC::Detail::TaskDesc desc;
    desc.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    desc.push_back(UC::Detail::Shard{blockId, 0, {dst.data()}});
    auto task =
        std::make_shared<TransTask>(TransTask::Type::LOAD_ON_TOKEN, std::move(desc), filter);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_TRUE(failureSet.Contains(task->id));
    ASSERT_EQ(task->FailureStatus(), UC::Status::NotFound());
    EXPECT_TRUE(
        std::all_of(dst.begin(), dst.end(), [](std::byte b) { return b == std::byte{0}; }));
}

TEST_F(UCCacheLoadQueueTest, DdrLoadOnTokenLargeTaskSplicesInParallel)
{
    using namespace UC::CacheStore;
    constexpr size_t kTokenCount = 1024;
    constexpr size_t kTokenBytes = 512;
    constexpr size_t tensorSize = kTokenCount * kTokenBytes;
    std::vector<std::byte> payload(tensorSize);
    for (size_t i = 0; i < tensorSize; ++i) {
        payload[i] = static_cast<std::byte>((i * 31 + 7) % 251);
    }
    std::vector<std::byte> dst(tensorSize, std::byte{0});
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
        std::memcpy(task.front().addrs.front(), payload.data(), tensorSize);
        return NextId();
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = tensorSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("f1b2c3d4e5f6789012345678901234ab");
    std::vector<UC::Detail::TokenFilter> filter(1);
    filter[0].ranges.resize(1);
    for (size_t t = 0; t < kTokenCount; ++t) {
        filter[0].ranges[0].push_back(UC::Detail::TokenRange{t * kTokenBytes, kTokenBytes,
                                                            t * kTokenBytes});
    }

    UC::Detail::TaskDesc desc;
    desc.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    desc.push_back(UC::Detail::Shard{blockId, 0, {dst.data()}});
    auto task =
        std::make_shared<TransTask>(TransTask::Type::LOAD_ON_TOKEN, std::move(desc), filter);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitForDuration(5000));
    ASSERT_FALSE(failureSet.Contains(task->id));
    EXPECT_EQ(std::memcmp(dst.data(), payload.data(), tensorSize), 0);
}

TEST_F(UCCacheLoadQueueTest, DdrLoadOnTokenWaitsForSharedShardThenSplices)
{
    using namespace UC::CacheStore;
    constexpr size_t tensorSize = 4096;
    std::vector<std::byte> payload(tensorSize, std::byte{0xa5});
    std::vector<std::byte> localDst(tensorSize, std::byte{0});
    std::vector<std::byte> sharedDst(tensorSize, std::byte{0});
    UC::Test::Detail::MockStore backend;
    std::promise<void> backendSubmitted;
    auto backendSubmittedFuture = backendSubmitted.get_future().share();
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
        EXPECT_EQ(task.memoryType, UC::Detail::TaskDesc::MemoryType::DDR);
        std::memcpy(task.front().addrs.front(), payload.data(), tensorSize);
        backendSubmitted.set_value();
        return NextId();
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = tensorSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto localBlockId = UC::Test::Detail::TypesHelper::MakeBlockId("11b2c3d4e5f6789012345678901234ab");
    auto sharedBlockId = UC::Test::Detail::TypesHelper::MakeBlockId("22b2c3d4e5f6789012345678901234ab");

    auto sharedOwner = buffer.Get(sharedBlockId, 0, true, true);
    ASSERT_TRUE(sharedOwner);
    ASSERT_FALSE(sharedOwner.Ready());
    void* sharedSlot = sharedOwner.Data();

    std::thread marker([sharedOwner, sharedSlot, &payload, backendSubmittedFuture]() mutable {
        backendSubmittedFuture.wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::memcpy(sharedSlot, payload.data(), tensorSize);
        sharedOwner.MarkReady();
    });

    std::vector<UC::Detail::TokenFilter> filter(2);
    filter[0].ranges = {
        {UC::Detail::TokenRange{0, tensorSize, 0}}
    };
    filter[1].ranges = {
        {UC::Detail::TokenRange{0, tensorSize, 0}}
    };
    UC::Detail::TaskDesc desc;
    desc.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    desc.push_back(UC::Detail::Shard{localBlockId, 0, {localDst.data()}});
    desc.push_back(UC::Detail::Shard{sharedBlockId, 0, {sharedDst.data()}});
    auto task =
        std::make_shared<TransTask>(TransTask::Type::LOAD_ON_TOKEN, std::move(desc), filter);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    marker.join();
    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(task->id));
    EXPECT_EQ(std::memcmp(localDst.data(), payload.data(), tensorSize), 0);
    EXPECT_EQ(std::memcmp(sharedDst.data(), payload.data(), tensorSize), 0);
}

TEST_F(UCCacheLoadQueueTest, DdrLoadOnTokenAppliesTensorBaseOffsets)
{
    /* Cache-slot layout is [full K][full V][...]: a K token (tensor 0) lives at
     * slot + tensorOffset, a V token (tensor 1) lives at slot + tensorSizes[0]
     * + tensorOffset. A splice that misses the tensor base would read the V
     * range from the K region (0x11) instead of the V region (0x22). */
    using namespace UC::CacheStore;
    constexpr size_t kTensorBytes = 64;
    constexpr size_t vTensorBytes = 128;
    constexpr size_t shardSize = kTensorBytes + vTensorBytes;
    std::vector<std::byte> payload;
    payload.reserve(shardSize);
    payload.insert(payload.end(), kTensorBytes, std::byte{0x11});
    payload.insert(payload.end(), vTensorBytes, std::byte{0x22});
    std::vector<std::byte> dst(shardSize, std::byte{0});
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke([&](UC::Detail::TaskDesc task) {
        std::memcpy(task.front().addrs.front(), payload.data(), shardSize);
        return NextId();
    }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {kTensorBytes, vTensorBytes};
    config.shardSize = shardSize;
    config.blockSize = shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("41b2c3d4e5f6789012345678901234ab");
    std::vector<UC::Detail::TokenFilter> filter(1);
    filter[0].ranges.resize(2);
    filter[0].ranges[0].push_back(UC::Detail::TokenRange{16, 16, 0});
    filter[0].ranges[1].push_back(UC::Detail::TokenRange{32, 32, 16});

    UC::Detail::TaskDesc desc;
    desc.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    desc.push_back(UC::Detail::Shard{blockId, 0, {dst.data()}});
    auto task =
        std::make_shared<TransTask>(TransTask::Type::LOAD_ON_TOKEN, std::move(desc), filter);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(task->id));
    EXPECT_EQ(std::memcmp(dst.data(), payload.data() + 16, 16), 0);
    EXPECT_EQ(std::memcmp(dst.data() + 16, payload.data() + kTensorBytes + 32, 32), 0);
    EXPECT_TRUE(std::all_of(dst.begin() + 48, dst.end(),
                            [](std::byte b) { return b == std::byte{0}; }));
}

TEST_F(UCCacheLoadQueueTest, DdrLoadOnTokenInvalidFilterFailsInline)
{
    using namespace UC::CacheStore;
    constexpr size_t tensorSize = 4096;
    std::vector<std::byte> dst(tensorSize, std::byte{0});
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).Times(0);
    EXPECT_CALL(backend, Wait).Times(0);
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = tensorSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("31b2c3d4e5f6789012345678901234ab");

    UC::Detail::TaskDesc desc;
    desc.memoryType = UC::Detail::TaskDesc::MemoryType::DDR;
    desc.push_back(UC::Detail::Shard{blockId, 0, {dst.data()}});
    std::vector<UC::Detail::TokenFilter> emptyFilter;
    auto task =
        std::make_shared<TransTask>(TransTask::Type::LOAD_ON_TOKEN, std::move(desc), emptyFilter);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_TRUE(failureSet.Contains(task->id));
    EXPECT_TRUE(
        std::all_of(dst.begin(), dst.end(), [](std::byte b) { return b == std::byte{0}; }));
}
