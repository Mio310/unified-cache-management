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
#include <gtest/gtest.h>
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

TEST_F(UCCacheLoadQueueTest, LoadHostBufferBypassesCache)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    UC::Test::Detail::DataGenerator first{1, 4096};
    UC::Test::Detail::DataGenerator second{1, 4096};
    first.Generate();
    second.Generate();
    void* addr0 = first.Buffer();
    void* addr1 = second.Buffer();
    EXPECT_CALL(backend, Load)
        .WillOnce(testing::Invoke([&](UC::Detail::TaskDesc desc) -> UC::Expected<UC::Detail::TaskHandle> {
            EXPECT_EQ(desc.bufferSpace, UC::Detail::BufferSpace::HOST);
            EXPECT_EQ(desc.brief, "HostDirect");
            EXPECT_EQ(desc.size(), 1u);
            if (desc.size() == 1) {
                EXPECT_EQ(desc[0].addrs.size(), 2u);
                if (desc[0].addrs.size() == 2) {
                    EXPECT_EQ(desc[0].addrs[0], addr0);
                    EXPECT_EQ(desc[0].addrs[1], addr1);
                }
                EXPECT_EQ(desc[0].lengths.size(), 2u);
                if (desc[0].lengths.size() == 2) {
                    EXPECT_EQ(desc[0].lengths[0], 4096u);
                    EXPECT_EQ(desc[0].lengths[1], 4096u);
                }
            }
            return NextId();
        }));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {4096, 4096};
    config.shardSize = 8192;
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
    UC::Detail::TaskDesc desc{
        {blockId, 0, {addr0, addr1}}
    };
    desc.bufferSpace = UC::Detail::BufferSpace::HOST;
    auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, std::move(desc));
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(task->id));
    ASSERT_FALSE(buffer.Exist(blockId, 0));
}

TEST_F(UCCacheLoadQueueTest, PrefetchFillsCacheFromBackend)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load)
        .WillOnce(testing::Invoke([](UC::Detail::TaskDesc desc) -> UC::Expected<UC::Detail::TaskHandle> {
            EXPECT_EQ(desc.brief, "Backend2Cache");
            EXPECT_EQ(desc.size(), 1u);
            if (desc.size() == 1) {
                EXPECT_EQ(desc[0].addrs.size(), 1u);
                if (!desc[0].addrs.empty()) { EXPECT_NE(desc[0].addrs[0], nullptr); }
            }
            return NextId();
        }));
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
    UC::Detail::TaskDesc desc;
    UC::Detail::Shard shard;
    shard.owner = blockId;
    shard.index = 0;
    desc.push_back(std::move(shard));
    auto task = std::make_shared<TransTask>(TransTask::Type::PREFETCH, std::move(desc));
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(task->id));
    ASSERT_TRUE(buffer.Exist(blockId, 0));
    auto cached = buffer.Get(blockId, 0, true, true);
    ASSERT_TRUE(cached.Ready());

    UC::Detail::TaskDesc againDesc;
    UC::Detail::Shard again;
    again.owner = blockId;
    again.index = 0;
    againDesc.push_back(std::move(again));
    auto second = std::make_shared<TransTask>(TransTask::Type::PREFETCH, std::move(againDesc));
    auto waiter2 = std::make_shared<UC::Latch>();
    loadQ.Submit(second, waiter2);
    ASSERT_TRUE(waiter2->WaitForDuration(1000));
    ASSERT_FALSE(failureSet.Contains(second->id));
}
