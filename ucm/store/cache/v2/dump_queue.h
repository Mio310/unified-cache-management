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
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "cache_buffer.h"
#include "copy_stream.h"
#include "ctrl_layout.h"
#include "global_config.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "status/status.h"
#include "template/spsc_ring_queue.h"
#include "thread/cpu_affinity.h"
#include "time/now_time.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::Cache2 {

/**
 * @brief Dump pipeline: device KV shards -> host cache buffer -> backend store.
 *
 * Two-stage design on SPSC queues (single producer: TransManager dispatch):
 *  - dispatcher_: pop task -> wait prerequisite event -> pin buffer slots ->
 *    D2H gather -> mark slots ready -> submit to backend -> signal waiter.
 *    A task is complete for the caller once its data is in the cache buffer
 *    and the backend write has been submitted.
 *  - dumper_: wait for each backend write, then release the pinned slots.
 * Shards whose slot is owned by a concurrent task are skipped (dedup).
 */
template <typename BufferT = Buffer, typename StreamT = CopyStream>
class DumpQ {
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using Handle = typename BufferT::Handle;
    using SlotState = CtrlLayout::SlotMeta::State;
    struct DumpCtx {
        Detail::TaskHandle taskHandle{};
        Detail::TaskHandle backendTaskHandle{};
        std::vector<Handle> bufferHandles{};
        TaskPtr task{};
        WaiterPtr waiter{};
    };

    alignas(64) std::atomic_bool stop_{false};
    alignas(64) std::atomic_bool dumperStop_{false};
    TaskIdSet* failureSet_{nullptr};
    BufferT* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    size_t streamNumber_{1};
    bool sdmaDirect_{false};
    bool ioAggregation_{false};
    bool useGdr_{false};
    bool waitForBackend_{false};
    std::vector<size_t> tensorSizes_{};
    std::vector<ssize_t> cpuAffinityCores_{};
    size_t timeoutMs_{0};
    std::mutex submitMutex_{};
    std::mutex closeMutex_{};
    std::mutex activeMutex_{};
    TaskPtr activeTask_{};
    SpscRingQueue<TaskPair> waiting_{};
    SpscRingQueue<DumpCtx> dumping_{};
    std::thread dispatcher_{};
    std::thread dumper_{};

public:
    DumpQ() = default;
    ~DumpQ() { Close(); }
    DumpQ(const DumpQ&) = delete;
    DumpQ& operator=(const DumpQ&) = delete;

    Status Setup(const Config& config, TaskIdSet* failureSet, BufferT* buffer)
    {
        failureSet_ = failureSet;
        buffer_ = buffer;
        backend_ = config.storeBackend;
        deviceId_ = config.deviceId;
        streamNumber_ = config.EffectiveStreamNumber();
        sdmaDirect_ = config.sdmaDirect;
        ioAggregation_ = config.ioAggregation;
        useGdr_ = config.useGdr;
        waitForBackend_ = config.enableDramTailEviction;
        tensorSizes_ = config.tensorSizes;
        cpuAffinityCores_ = config.cpuAffinityCores;
        timeoutMs_ = config.timeoutMs;
        waiting_.Setup(config.waitingQueueDepth);
        if (backend_ != nullptr) {
            dumping_.Setup(config.runningQueueDepth);
            dumper_ = std::thread{&DumpQ::BackendWaitStage, this};
        }
        std::promise<Status> started;
        auto fut = started.get_future();
        dispatcher_ = std::thread{&DumpQ::DispatchStage, this, std::ref(started)};
        return fut.get();
    }
    void Submit(TaskPtr task, WaiterPtr waiter)
    {
        waiter->Up();
        bool stopped = false;
        {
            std::lock_guard<std::mutex> lock(submitMutex_);
            stopped = stop_.load(std::memory_order_acquire);
            if (!stopped && waiting_.TryPush({task, waiter})) { return; }
        }
        if (stopped) {
            task->Fail(Status::Error("dump queue closed"));
            task->relationLease.reset();
            failureSet_->Insert(task->id);
            waiter->Done();
            return;
        }
        UC_ERROR("Waiting queue full, submit dump task({}) failed.", task->id);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_full_total"), 1.0);
        task->Fail(Status::Retry());
        task->relationLease.reset();
        failureSet_->Insert(task->id);
        waiter->Done();
    }
    void Close()
    {
        std::lock_guard<std::mutex> closeLock(closeMutex_);
        {
            std::lock_guard<std::mutex> submitLock(submitMutex_);
            if (stop_.exchange(true)) { return; }
        }
        {
            std::lock_guard<std::mutex> lock(activeMutex_);
            if (activeTask_) { activeTask_->cancelled.store(true, std::memory_order_release); }
        }
        if (dispatcher_.joinable()) { dispatcher_.join(); }
        DrainWaiting();
        // Keep the backend consumer alive while the dispatcher may be blocked
        // pushing a completed write into the bounded queue.
        dumperStop_.store(true, std::memory_order_release);
        if (dumper_.joinable()) { dumper_.join(); }
        DrainDumping();
    }

private:
    void DispatchStage(std::promise<Status>& started)
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_dump_d2h");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set dump d2h thread name.", nameStatus);
        }
        StreamT stream;
        auto s = Status::OK();
        if (ioAggregation_) {
            s = stream.SetupIoAggregation(deviceId_, useGdr_);
        } else if (sdmaDirect_) {
            s = stream.SetupSdmaDirect(deviceId_, streamNumber_, useGdr_);
        } else {
            s = stream.Setup(deviceId_, streamNumber_, useGdr_);
        }
        started.set_value(s);
        if (s.Failure()) [[unlikely]] { return; }
        BindCpuAffinity();
        waiting_.ConsumerLoop(stop_, &DumpQ::DispatchOneTask, this, stream);
    }
    void DispatchOneTask(StreamT& stream, TaskPair&& pair)
    {
        auto& [task, waiter] = pair;
        auto queueWaitMs = (NowTime::Now() - waiter->startTp) * 1e3;
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_wait_duration_ms"), queueWaitMs);
        if (!failureSet_->Contains(task->id)) {
            {
                std::lock_guard<std::mutex> lock(activeMutex_);
                activeTask_ = task;
                if (stop_.load(std::memory_order_acquire)) {
                    task->cancelled.store(true, std::memory_order_release);
                }
            }
            auto s = DumpOneTask(stream, task, waiter);
            {
                std::lock_guard<std::mutex> lock(activeMutex_);
                activeTask_.reset();
            }
            if (s.Failure()) [[unlikely]] {
                task->Fail(s);
                failureSet_->Insert(task->id);
            }
        }
        task->relationLease.reset();
        waiter->Done();
    }
    Status DumpOneTask(StreamT& stream, const TaskPtr& task, const WaiterPtr& waiter)
    {
        auto s = WaitPrerequisite(stream, task);
        if (s.Failure()) [[unlikely]] { return s; }
        DumpCtx ctx;
        ctx.taskHandle = task->id;
        ctx.bufferHandles.reserve(task->desc.size());
        Detail::TaskDesc backendDesc;
        backendDesc.brief = "Cache2Backend";
        size_t copiedShards = 0;
        s = GatherShards(stream, task, ctx, backendDesc, copiedShards);
        if (s.Success() && task->cancelled.load(std::memory_order_acquire)) { s = Status::Retry(); }
        if (s.Failure()) [[unlikely]] {
            // A previous shard may still be the target of an asynchronous copy.
            // Drain every copy stream before changing state or releasing handles.
            if (!ctx.bufferHandles.empty()) {
                auto syncStatus = stream.Synchronize();
                if (syncStatus.Failure()) {
                    UC_ERROR("Failed({}) to drain D2H after dump task({}) failed.", syncStatus,
                             task->id);
                    QuarantineHandles(ctx);
                    FailStopOnUndrainedDma();
                } else {
                    FailOwnedHandles(ctx);
                }
            }
            return s;
        }
        if (ctx.bufferHandles.empty()) { return Status::OK(); }
        s = CompleteD2h(stream, task, ctx, copiedShards);
        if (s.Failure()) [[unlikely]] { return s; }
        if (task->cancelled.load(std::memory_order_acquire)) { return Status::Retry(); }
        if (backend_ == nullptr || backendDesc.empty()) { return Status::OK(); }
        return SubmitBackendDump(task, ctx, std::move(backendDesc), waiter);
    }
    Status WaitPrerequisite(StreamT& stream, const TaskPtr& task)
    {
        if (task->desc.prerequisiteHandle == 0) { return Status::OK(); }
        auto s = stream.WaitEvent(Trans::Event{task->desc.prerequisiteHandle});
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait prerequisite event for dump task({}).", s, task->id);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
        }
        return s;
    }
    Status GatherShards(StreamT& stream, const TaskPtr& task, DumpCtx& ctx,
                        Detail::TaskDesc& backendDesc, size_t& copiedShards)
    {
        const auto startTp = NowTime::Now();
        // Always bound the allocation wait, including fire-and-forget callers.
        constexpr size_t kDefaultAllocationTimeoutMs = 30000;
        const auto allocationTimeout = timeoutMs_ == 0 ? kDefaultAllocationTimeoutMs : timeoutMs_;
        const auto now = std::chrono::steady_clock::now();
        const auto maxWait = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::time_point::max() - now).count();
        const auto timeout = std::min(allocationTimeout, static_cast<size_t>(maxWait));
        const auto deadline = now + std::chrono::milliseconds(timeout);
        std::vector<Detail::Shard*> uniqueShards;
        try {
            if (backend_ != nullptr) { backendDesc.reserve(task->desc.size()); }
            if (waitForBackend_) {
                std::unordered_map<Detail::BlockId, std::unordered_set<size_t>, Detail::BlockIdHasher>
                    seenShards;
                uniqueShards.reserve(task->desc.size());
                for (auto& shard : task->desc) {
                    if (seenShards[shard.owner].insert(shard.index).second) {
                        uniqueShards.push_back(&shard);
                    }
                }
            }
        } catch (const std::bad_alloc&) {
            return Status::OutOfMemory();
        }
        const auto shardCount = waitForBackend_ ? uniqueShards.size() : task->desc.size();
        for (size_t i = 0; i < shardCount; ++i) {
            // Deduplicate before starting any device copy: waiting on our own
            // in-flight copy would otherwise deadlock until the deadline.
            auto& shard = waitForBackend_ ? *uniqueShards[i] : task->desc[i];
            auto result = buffer_->GetUntil(shard.owner, shard.index, deadline, task->cancelled);
            if (!result) { return result.Error(); }
            auto handle = std::move(result).Value();
            if (task->cancelled.load(std::memory_order_acquire)) {
                if (handle.Owner() && handle.GetState() != SlotState::Ready) { handle.MarkFailed(); }
                return Status::Retry();
            }
            if (!handle.Owner()) {
                if (!waitForBackend_) { continue; }
                // Another task owns this slot. Its device copy and backend
                // dump may still fail, so a duplicate cannot report success
                // merely because the key exists. Keep the slot pinned, wait
                // for valid contents, and persist them as part of this task.
                ctx.bufferHandles.push_back(std::move(handle));
                auto& existing = ctx.bufferHandles.back();
                for (;;) {
                    const auto state = existing.GetState();
                    if (state == SlotState::Ready) { break; }
                    if (state == SlotState::Failed) { return Status::Error("concurrent dump failed"); }
                    if (task->cancelled.load(std::memory_order_acquire)) { return Status::Retry(); }
                    if (std::chrono::steady_clock::now() >= deadline) { return Status::Timeout(); }
                    std::this_thread::yield();
                }
                if (backend_ != nullptr && existing.HostAccessible()) {
                    backendDesc.push_back(Detail::Shard{shard.owner, shard.index, {existing.Data()}});
                }
                continue;
            }
            ctx.bufferHandles.push_back(std::move(handle));
            auto& pinned = ctx.bufferHandles.back();
            const auto hostAccessible = pinned.HostAccessible();
            if (pinned.GetState() != SlotState::Ready) {
                /* SDMA Direct feeds device descriptors, so the host side has to
                 * be the device-visible alias rather than the CPU address. */
                void* hostDst = sdmaDirect_ ? pinned.HostMappedData() : pinned.Data();
                auto s =
                    hostAccessible
                        ? stream.DeviceToHostGatherAsync(shard.addrs.data(), hostDst, tensorSizes_)
                        : stream.DeviceToDeviceGatherAsync(shard.addrs.data(), pinned.DeviceData(),
                                                           tensorSizes_);
                if (s.Failure()) [[unlikely]] {
                    UC_ERROR("Failed({}) to copy shard for dump task({}).", s, task->id);
                    Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
                    return s;
                }
                ++copiedShards;
            }
            if (backend_ != nullptr && hostAccessible) {
                backendDesc.push_back(Detail::Shard{shard.owner, shard.index, {pinned.Data()}});
            }
        }
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_mkbuf_duration_ms"),
                             (NowTime::Now() - startTp) * 1e3);
        return Status::OK();
    }
    Status CompleteD2h(StreamT& stream, const TaskPtr& task, DumpCtx& ctx, size_t copiedShards)
    {
        if (copiedShards > 0) {
            const auto startTp = NowTime::Now();
            auto s = stream.Synchronize();
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to sync stream for dump task({}).", s, task->id);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
                // A failed synchronization does not prove the device stopped
                // writing. Keep every affected slot pinned for this domain.
                QuarantineHandles(ctx);
                FailStopOnUndrainedDma();
                return s;
            }
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_duration_ms"),
                                 (NowTime::Now() - startTp) * 1e3);
        }
        for (auto& handle : ctx.bufferHandles) { handle.MarkReady(); }
        return Status::OK();
    }
    Status SubmitBackendDump(const TaskPtr& task, DumpCtx& ctx, Detail::TaskDesc&& backendDesc,
                             const WaiterPtr& waiter)
    {
        const auto startTp = NowTime::Now();
        const auto nBackendShards = backendDesc.size();
        auto res = backend_->Dump(std::move(backendDesc));
        if (!res) [[unlikely]] {
            auto error = res.Error();
            if (error != Status::StoreUnhealthy()) {
                UC_ERROR("Failed({}) to submit dump task({}) to backend.", error, task->id);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_submit_errors_total"),
                                     1.0);
            }
            return error;
        }
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_shards_total"),
                             static_cast<double>(nBackendShards));
        ctx.backendTaskHandle = res.Value();
        if (waitForBackend_) {
            // The caller must not observe success until the backend has
            // finished: in tail mode it may be the only copy after eviction.
            ctx.task = task;
            ctx.waiter = waiter;
            waiter->Up();
        }
        dumping_.Push(std::move(ctx));
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_submit_duration_ms"),
                             (NowTime::Now() - startTp) * 1e3);
        return Status::OK();
    }
    void FailOwnedHandles(DumpCtx& ctx)
    {
        for (auto& handle : ctx.bufferHandles) { handle.MarkFailed(); }
    }
    void QuarantineHandles(DumpCtx& ctx)
    {
        for (auto& handle : ctx.bufferHandles) { handle.Quarantine(); }
    }
    void FailStopOnUndrainedDma() const
    {
        if (!waitForBackend_) { return; }
        // A failed synchronization leaves the caller's device buffers as well
        // as the DRAM slots potentially in use. Returning the task would let
        // vLLM recycle those buffers, so stop this worker rather than claim
        // that the failed transfer is complete.
        UC_ERROR("cache2 cannot establish DMA completion; terminating worker");
        std::terminate();
    }
    void BackendWaitStage()
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_dump_wait");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set dump wait thread name.", nameStatus);
        }
        BindCpuAffinity();
        dumping_.ConsumerLoop(dumperStop_, [this](DumpCtx&& ctx) {
            if (ctx.backendTaskHandle == 0) {
                CompleteBackend(ctx, Status::OK());
                return;
            }
            const auto startTp = NowTime::Now();
            auto s = backend_->Wait(ctx.backendTaskHandle);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_wait_duration_ms"),
                                 (NowTime::Now() - startTp) * 1e3);
            if (s.Failure()) [[unlikely]] {
                if (s != Status::StoreUnhealthy()) {
                    UC_ERROR("Failed({}) to wait backend({}) for dump task({}).", s,
                             ctx.backendTaskHandle, ctx.taskHandle);
                }
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_wait_errors_total"),
                                     1.0);
            }
            CompleteBackend(ctx, s);
        });
    }
    void CompleteBackend(DumpCtx& ctx, const Status& status)
    {
        if (!ctx.waiter) { return; }
        if (status.Failure()) {
            ctx.task->Fail(status);
            failureSet_->Insert(ctx.task->id);
        }
        ctx.waiter->Done();
        ctx.waiter.reset();
        ctx.task.reset();
    }
    void DrainWaiting()
    {
        TaskPair pair;
        while (waiting_.TryPop(pair)) {
            UC_WARN("Cache2 dump task({}) discarded on close.", pair.first->id);
            pair.first->Fail(Status::Error("dump queue closed"));
            pair.first->relationLease.reset();
            failureSet_->Insert(pair.first->id);
            pair.second->Done();
        }
    }
    void BindCpuAffinity() const
    {
        if (cpuAffinityCores_.empty()) { return; }
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set cache dump affinity.", s); }
    }
    void DrainDumping()
    {
        if (backend_ == nullptr) { return; }
        DumpCtx ctx;
        while (dumping_.TryPop(ctx)) {
            if (ctx.backendTaskHandle == 0) {
                CompleteBackend(ctx, Status::OK());
                continue;
            }
            auto s = backend_->Wait(ctx.backendTaskHandle);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to wait backend({}) for dump task({}).", s,
                         ctx.backendTaskHandle, ctx.taskHandle);
            }
            CompleteBackend(ctx, s);
        }
    }
};

}  // namespace UC::Cache2
