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
#include "load_queue.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

LoadQueue::~LoadQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    nShardPerBlock_ = config.blockSize / config.shardSize;
    streamNumber_ = config.EffectiveStreamNumber();
    useGdr_ = config.useGdr;
    cacheIOAggregation_ = config.cacheIOAggregation;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    cpuAffinityCores_ = config.cpuAffinityCores;
    localRankSize_ = config.localRankSize;
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    transfer_ = std::thread{&LoadQueue::TransferStage, this, std::ref(started)};
    return fut.get();
}

void LoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_full_total"), 1.0);
    RecordFailedShards(task->desc.size());
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_disp");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM load dispatcher name.", nameStatus);
    }
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
}

static std::vector<size_t> RearrangeIndex(size_t n, size_t iProc, size_t nProc)
{
    std::vector<size_t> order;
    order.reserve(n);
    for (size_t r = 0; r < nProc; ++r) {
        size_t slice = (iProc + r) % nProc;
        for (size_t j = 0;; ++j) {
            size_t i = slice + j * nProc;
            if (i >= n) { break; }
            order.push_back(i);
        }
    }
    return order;
}

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    if (task->desc.bufferSpace == Detail::BufferSpace::HOST) {
        DispatchHostTask(task, waiter);
        return;
    }
    auto tp = waiter->startTp;
    auto tpWait = NowTime::Now();
    const auto nShard = task->desc.size();
    size_t backendSubmitCount = 0;
    size_t waitShardCount = 0;
    const auto indexes = RearrangeIndex(nShard, deviceId_, localRankSize_);
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[indexes[i]];
        ShardTask shardTask;
        shardTask.bufferHandle = buffer_->Get(shard.owner, shard.index, true, true);
        shardTask.backendTaskHandle = 0;
        shardTask.fromPosix = !shardTask.bufferHandle.Ready();
        if (shardTask.fromPosix) { waitShardCount++; }
        if (shardTask.bufferHandle.Owner() && !shardTask.bufferHandle.Ready()) {
            Detail::TaskDesc backendTask{
                Detail::Shard{shard.owner, shard.index, {shardTask.bufferHandle.Data()}}
            };
            backendTask.brief = "Backend2Cache";
            auto res = backend_->Load(std::move(backendTask));
            if (!res) [[unlikely]] {
                UC_ERROR("Failed({}) to submit load task({}) to backend.", res.Error(), task->id);
                UC::Metrics::UpdateStats(
                    NAME_TO_METRIC_ID("cache_backend_load_submit_errors_total"), 1.0);
                RecordLoadSourceShards(i + 1, waitShardCount);
                RecordFailedShards(nShard - i);
                shardTask.bufferHandle.MarkFailed(res.Error());
                task->Fail(res.Error());
                failureSet_->Insert(task->id);
                waiter->Done();
                return;
            }
            shardTask.backendTaskHandle = res.Value();
            backendSubmitCount++;
        }
        shardTask.task = task;
        shardTask.shard = std::move(shard);
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
        running_.Push(std::move(shardTask));
    }
    auto tpDispatch = NowTime::Now();
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[indexes[i]];
        if (shard.index + 1 != nShardPerBlock_) {
            buffer_->Prealloc(shard.owner, shard.index + 1, true);
        }
    }
    UC_DEBUG("Cache task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
             (tpWait - tp) * 1e3, (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"),
                             (tpWait - tp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_submit_duration_ms"),
                             (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
    RecordLoadSourceShards(nShard, waitShardCount);
}

Status LoadQueue::BuildHostBackend(const Detail::TaskDesc& desc, Detail::TaskDesc& backend) const
{
    backend.brief = "HostDirect";
    backend.bufferSpace = Detail::BufferSpace::HOST;
    backend.reserve(desc.size());
    for (const auto& shard : desc) {
        if (shard.addrs.empty() || shard.addrs.front() == nullptr) {
            return Status::InvalidParam("host load shard({}) has no address", shard.index);
        }
        Detail::Shard out;
        out.owner = shard.owner;
        out.index = shard.index;
        out.addrs = shard.addrs;
        if (!shard.lengths.empty()) {
            if (shard.lengths.size() != shard.addrs.size()) {
                return Status::InvalidParam("host load length count({}) != address count({})",
                                            shard.lengths.size(), shard.addrs.size());
            }
            out.lengths = shard.lengths;
        } else if (shard.addrs.size() > 1) {
            if (shard.addrs.size() != tensorSizes_.size()) {
                return Status::InvalidParam("host load address count({}) != tensor count({})",
                                            shard.addrs.size(), tensorSizes_.size());
            }
            out.lengths = tensorSizes_;
        }
        backend.push_back(std::move(out));
    }
    return Status::OK();
}

void LoadQueue::DispatchHostTask(const TaskPtr& task, const WaiterPtr& waiter)
{
    auto tp = waiter->startTp;
    auto tpWait = NowTime::Now();
    const auto nShard = task->desc.size();
    Detail::TaskDesc backendTask;
    auto built = BuildHostBackend(task->desc, backendTask);
    if (built.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to build host load task({}).", built, task->id);
        RecordFailedShards(nShard);
        task->Fail(built);
        failureSet_->Insert(task->id);
        waiter->Done();
        return;
    }
    Detail::TaskHandle backendHandle = 0;
    if (!backendTask.empty()) {
        auto res = backend_->Load(std::move(backendTask));
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit host load task({}) to backend.", res.Error(), task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_submit_errors_total"),
                                     1.0);
            RecordLoadSourceShards(nShard, nShard);
            RecordFailedShards(nShard);
            task->Fail(res.Error());
            failureSet_->Insert(task->id);
            waiter->Done();
            return;
        }
        backendHandle = res.Value();
    }
    ShardTask shardTask;
    shardTask.task = task;
    shardTask.backendTaskHandle = backendHandle;
    shardTask.waiter = waiter;
    shardTask.fromPosix = true;
    shardTask.hostDirect = true;
    running_.Push(std::move(shardTask));
    auto tpDispatch = NowTime::Now();
    UC_DEBUG("Cache host task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id,
             nShard, (tpWait - tp) * 1e3, (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"),
                             (tpWait - tp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_submit_duration_ms"),
                             (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(nShard));
    RecordLoadSourceShards(nShard, nShard);
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_xfer");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM load transfer name.", nameStatus); }
    CopyStream stream;
    auto s = Status::OK();
    if (cacheIOAggregation_) {
        s = stream.SetupIoAggregation(deviceId_, useGdr_);
    } else if (cacheSdmaDirect_) {
        s = stream.SetupSdmaDirect(deviceId_, useGdr_);
    } else {
        s = stream.Setup(deviceId_, streamNumber_, useGdr_);
    }
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, stream);
}

void LoadQueue::TransferHostTask(ShardTask&& task)
{
    auto parentTask = task.task;
    const auto taskHandle = parentTask->id;
    const auto nShard = parentTask->desc.size();
    auto s = Status::OK();
    if (failureSet_->Contains(taskHandle)) {
        s = parentTask->FailureStatus();
    } else if (task.backendTaskHandle != 0) {
        auto tpBackendWait = NowTime::Now();
        s = backend_->Wait(task.backendTaskHandle);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 (NowTime::Now() - tpBackendWait) * 1e3);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for host task({}).", s, task.backendTaskHandle,
                     taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_wait_errors_total"), 1.0);
        }
    }
    if (s.Failure()) [[unlikely]] {
        parentTask->Fail(s);
        failureSet_->Insert(taskHandle);
        RecordFailedShards(nShard);
    } else {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_posix_load_success_shards_total"),
                                 static_cast<double>(nShard));
    }
    if (task.waiter) { task.waiter->Done(); }
}

void LoadQueue::TransferOneTask(CopyStream& stream, ShardTask&& task)
{
    if (task.hostDirect) {
        TransferHostTask(std::move(task));
        return;
    }
    auto parentTask = task.task;
    const auto taskHandle = parentTask->id;
    if (failureSet_->Contains(taskHandle)) {
        RecordFailedShards(1);
        if (task.waiter) {
            holder_.clear();
            task.waiter->Done();
        }
        return;
    }

    auto s = Status::OK();
    auto waiter = task.waiter;
    do {
        auto tpBackendWait = NowTime::Now();
        s = WaitBackendTaskReady(task);
        if (s.Failure()) [[unlikely]] {
            RecordShardResults(holder_, &task, false);
            break;
        }
        auto tpBackendReady = NowTime::Now();
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 (tpBackendReady - tpBackendWait) * 1e3);

        auto* host = cacheSdmaDirect_ ? task.bufferHandle.DeviceData() : task.bufferHandle.Data();
        s = HostToDeviceAsync(stream, host, task.shard.addrs.data());
        auto tpH2dSubmitted = NowTime::Now();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            RecordShardResults(holder_, &task, false);
            break;
        }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                                 (tpH2dSubmitted - tpBackendReady) * 1e3);
        if (!waiter) {
            holder_.push_back(std::move(task));
            return;
        }
        auto tpH2dSyncStart = NowTime::Now();
        s = stream.Synchronize();
        auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
        RecordH2dSyncMetrics(h2dSyncMs);
        RecordShardResults(holder_, &task, s.Success());
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
    } while (0);
    if (s.Failure()) [[unlikely]] {
        parentTask->Fail(s);
        failureSet_->Insert(taskHandle);
    }
    if (waiter) { waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.backendTaskHandle != 0) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_wait_errors_total"),
                                     1.0);
            task.bufferHandle.MarkFailed(s);
            return s;
        }
        task.bufferHandle.MarkReady();
        return Status::OK();
    }
    for (;;) {
        auto state = task.bufferHandle.GetState();
        if (state == TransBuffer::State::READY) { return Status::OK(); }
        if (state == TransBuffer::State::FAILED) { return task.bufferHandle.FailureStatus(); }
        if (failureSet_->Contains(task.task->id)) { return task.task->FailureStatus(); }
        std::this_thread::yield();
    }
}

Status LoadQueue::HostToDeviceAsync(CopyStream& stream, void* host, void** device)
{
    return stream.HostToDeviceAsync(host, device, tensorSizes_);
}

void LoadQueue::RecordShardResults(const std::vector<ShardTask>& tasks, const ShardTask* extra,
                                   bool success) const
{
    size_t cache = 0;
    size_t posix = 0;
    for (const auto& task : tasks) {
        if (task.fromPosix) {
            ++posix;
        } else {
            ++cache;
        }
    }
    if (extra != nullptr) {
        if (extra->fromPosix) {
            ++posix;
        } else {
            ++cache;
        }
    }
    if (!success) {
        RecordFailedShards(cache + posix);
        return;
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_success_shards_total"),
                             static_cast<double>(cache));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_posix_load_success_shards_total"),
                             static_cast<double>(posix));
}

void LoadQueue::RecordFailedShards(size_t count) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_failed_shards_total"),
                             static_cast<double>(count));
}

void LoadQueue::RecordLoadSourceShards(size_t total, size_t wait) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_shards_total"),
                             static_cast<double>(total));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_wait_shards_total"),
                             static_cast<double>(wait));
}

void LoadQueue::RecordH2dSyncMetrics(double h2dSyncMs) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_sync_ms"), h2dSyncMs);
}

}  // namespace UC::CacheStore
