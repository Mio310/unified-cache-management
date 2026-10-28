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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_IO_ENGINE_AIO_H
#define UNIFIEDCACHE_POSIX_STORE_CC_IO_ENGINE_AIO_H

#include <atomic>
#include <cerrno>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "aio_impl.h"
#include "block_operator.h"
#include "block_tracker.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::PosixStore {

class IoEngineAio : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    static constexpr size_t kWatchdogGraceMs = 2000;
    struct Inflight {
        TaskPtr t;
        WaiterPtr w;
        double deadlineTp{0};
        size_t shardCount{0};
        bool aborted{false};
    };
    struct HostReadGroup {
        TaskPtr task;
        WaiterPtr waiter;
        Detail::BlockId block{};
        int32_t fd{-1};
        std::atomic<size_t> remaining{0};
    };
    struct IoSeg {
        void* buffer{nullptr};
        size_t length{0};
        uint64_t offset{0};
    };
    size_t shardSize_;
    size_t ioSize_{0};
    size_t nShardPerBlock_;
    const SpaceLayout* layout_;
    BlockCommitTracker tracker_;
    std::mutex regMutex_;
    std::unordered_map<Detail::TaskHandle, Inflight> registry_;
    BlockOperator blockOperator_;
    AioImpl aio_;

public:
    // aio_ is destroyed before blockOperator_, but open callbacks submit I/O to aio_ and
    // completion callbacks submit commits to blockOperator_; stop the open/commit workers first.
    ~IoEngineAio() { blockOperator_.Stop(); }
    Status Setup(const Config& config, const SpaceLayout* layout)
    {
        timeoutMs_ = config.timeoutMs;
        shardSize_ = config.shardSize;
        ioSize_ = config.tensorSize;
        nShardPerBlock_ = config.blockSize / config.shardSize;
        layout_ = layout;
        tracker_.Setup(nShardPerBlock_);
        blockOperator_.Setup(layout, config.openConcurrency, config.commitConcurrency);
        aio_.SetSweepFn([this] { SweepDeadlines(); });
        UC_INFO(
            "AIO engine setup: timeoutMs={}, watchdogGraceMs={}, "
            "openConcurrency={}, commitConcurrency={}.",
            timeoutMs_, kWatchdogGraceMs, config.openConcurrency, config.commitConcurrency);
        return aio_.Setup(timeoutMs_);
    }

private:
    bool IsAio() const override { return true; }
    Status FailureStatus(const TaskPtr& task) const override { return task->FailureStatus(); }
    template <bool dump>
    static void UpdateWaitMetrics(double wait)
    {
        static UC::Metrics::CachedMetric dumpQueueDuration{"posix_dump_queue_wait_duration_ms"};
        static UC::Metrics::CachedMetric loadQueueDuration{"posix_load_queue_wait_duration_ms"};
        UC::Metrics::UpdateStats(dump ? dumpQueueDuration : loadQueueDuration, wait * 1e3);
    }
    static void IncrementAioTimeoutMetric()
    {
        static UC::Metrics::CachedMetric timeoutTotal{"posix_aio_timeout_total"};
        UC::Metrics::UpdateStats(timeoutTotal, 1.0);
    }
    static void IncrementOpenErrorMetric()
    {
        static UC::Metrics::CachedMetric openErrors{"posix_open_errors_total"};
        UC::Metrics::UpdateStats(openErrors, 1.0);
    }
    static void IncrementIoErrorMetric()
    {
        static UC::Metrics::CachedMetric ioErrors{"posix_io_errors_total"};
        UC::Metrics::UpdateStats(ioErrors, 1.0);
    }
    void CommitBlock(Detail::BlockId id, bool success)
    {
        blockOperator_.Submit(BlockOperator::CommitTask{std::move(id), success});
    }
    template <bool dump>
    void OnIoCallback(const TaskPtr& task, WaiterPtr w, int32_t fd, size_t shardIndex, uint64_t ino,
                      const Detail::BlockId& id, const AioImpl::Result& result)
    {
        const auto tid = task->id;
        const auto shortIo = result.error == 0 && result.nBytes != static_cast<ssize_t>(shardSize_);
        if (result.error != 0 || shortIo) {
            UC_ERROR("Failed(error={}, bytes={}/{}) to do io on block({}).", result.error,
                     result.nBytes, shardSize_, id);
            if (result.error != ECANCELED) { IncrementIoErrorMetric(); }
            task->Fail(!dump && shortIo ? Status::NotFound() : Status::Error());
            failureSet_.Insert(tid);
        }
        ::close(fd);
        if constexpr (dump) {
            const auto ok = result.error == 0 && !shortIo && !failureSet_.Contains(tid);
            if (ok && tracker_.MarkDone(id, shardIndex, ino)) {
                const auto path = layout_->DataFilePath(id, true);
                if (ino != 0 && InoOf(path) == ino) { CommitBlock(id, true); }
            } else if (!ok && shardIndex + 1 == nShardPerBlock_) {
                CommitBlock(id, false);
            }
        }
        w->Done();
    }
    Status BuildHostLoadSegs(const Detail::Shard& shard, std::vector<IoSeg>& segs) const
    {
        if (shard.addrs.empty()) { return Status::InvalidParam("host load shard has no address"); }
        const auto base = static_cast<uint64_t>(shard.index) * shardSize_;
        if (!shard.lengths.empty()) {
            if (shard.lengths.size() != shard.addrs.size()) {
                return Status::InvalidParam("host load length count({}) != address count({})",
                                            shard.lengths.size(), shard.addrs.size());
            }
            uint64_t off = base;
            size_t total = 0;
            segs.reserve(shard.addrs.size());
            for (size_t i = 0; i < shard.addrs.size(); ++i) {
                const auto len = shard.lengths[i];
                if (len == 0 || shard.addrs[i] == nullptr) {
                    return Status::InvalidParam("host load segment is empty");
                }
                if (len > std::numeric_limits<uint32_t>::max()) {
                    return Status::InvalidParam("host load segment is too large");
                }
                total += len;
                if (total > shardSize_) {
                    return Status::InvalidParam("host load segments exceed shard size({})",
                                                shardSize_);
                }
                segs.push_back(IoSeg{shard.addrs[i], len, off});
                off += len;
            }
            return Status::OK();
        }
        if (ioSize_ == 0 || ioSize_ > std::numeric_limits<uint32_t>::max() ||
            ioSize_ > shardSize_) {
            return Status::InvalidParam("invalid host load io size({})", ioSize_);
        }
        if (shard.addrs.size() > shardSize_ / ioSize_) {
            return Status::InvalidParam("host load address count({}) exceeds shard",
                                        shard.addrs.size());
        }
        uint64_t off = base;
        segs.reserve(shard.addrs.size());
        for (auto* addr : shard.addrs) {
            if (addr == nullptr) { return Status::InvalidParam("host load segment is empty"); }
            segs.push_back(IoSeg{addr, ioSize_, off});
            off += ioSize_;
        }
        return Status::OK();
    }
    void FinishHostSeg(const std::shared_ptr<HostReadGroup>& group, size_t expect,
                       const AioImpl::Result& result)
    {
        const auto shortIo = result.error == 0 && result.nBytes != static_cast<ssize_t>(expect);
        if (result.error != 0 || shortIo) {
            UC_ERROR("Failed(error={}, bytes={}/{}) to do host load on block({}).", result.error,
                     result.nBytes, expect, group->block);
            if (result.error != ECANCELED) { IncrementIoErrorMetric(); }
            group->task->Fail(shortIo ? Status::NotFound() : Status::Error());
            failureSet_.Insert(group->task->id);
        }
        if (group->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (group->fd >= 0) { ::close(group->fd); }
            group->waiter->Done();
        }
    }
    Status SubmitHostLoad(const TaskPtr& task, const WaiterPtr& w, int32_t fd,
                          const Detail::Shard& shard)
    {
        std::vector<IoSeg> segs;
        auto status = BuildHostLoadSegs(shard, segs);
        if (status.Failure() || segs.empty()) {
            if (status.Success()) {
                status = Status::InvalidParam("host load shard has no address");
            }
            return status;
        }
        auto group = std::make_shared<HostReadGroup>();
        group->task = task;
        group->waiter = w;
        group->block = shard.owner;
        group->fd = fd;
        group->remaining.store(segs.size(), std::memory_order_relaxed);
        size_t submitted = 0;
        for (const auto& seg : segs) {
            AioImpl::Io io;
            io.fd = fd;
            io.offset = seg.offset;
            io.length = static_cast<uint32_t>(seg.length);
            io.buffer = seg.buffer;
            io.tag = task->id;
            const auto expect = seg.length;
            io.callback = [this, group, expect](AioImpl::Result ioResult) {
                FinishHostSeg(group, expect, ioResult);
            };
            status = aio_.ReadAsync(std::move(io));
            if (status.Failure()) { break; }
            ++submitted;
        }
        if (submitted == segs.size()) { return Status::OK(); }
        task->Fail(status);
        failureSet_.Insert(task->id);
        if (status == Status::Timeout()) {
            IncrementAioTimeoutMetric();
        } else {
            IncrementIoErrorMetric();
        }
        const auto left = segs.size() - submitted;
        if (group->remaining.fetch_sub(left, std::memory_order_acq_rel) == left) {
            if (fd >= 0) { ::close(fd); }
            w->Done();
        }
        return Status::OK();
    }
    template <bool dump>
    void OnOpenCallback(const TaskPtr& task, WaiterPtr w, const Detail::Shard& shard,
                        const BlockOperator::OpenResult& result)
    {
        const auto tid = task->id;
        const auto shardIndex = shard.index;
        const auto last = shardIndex + 1 == nShardPerBlock_;
        const auto ino = dump ? InoOf(result.fd) : 0;
        const auto& id = shard.owner;
        auto handleFailure = [this, task, tid, w, last, id](int32_t fd, Status status) {
            task->Fail(status);
            failureSet_.Insert(tid);
            if (fd >= 0) { ::close(fd); }
            if constexpr (dump) {
                if (last) { CommitBlock(id, false); }
            }
            w->Done();
        };
        if (result.error != 0) {
            UC_ERROR("Failed({}) to do open on block({}).", result.error, shard.owner);
            if (result.error != ECANCELED) { IncrementOpenErrorMetric(); }
            auto status = !dump && result.error == ENOENT ? Status::NotFound() : Status::Error();
            handleFailure(result.fd, status);
            return;
        }
        if (failureSet_.Contains(tid)) {
            if (result.fd >= 0) { ::close(result.fd); }
            if constexpr (dump) {
                if (last) { CommitBlock(id, false); }
            }
            w->Done();
            return;
        }
        if constexpr (!dump) {
            if (task->desc.bufferSpace == Detail::BufferSpace::HOST) {
                auto status = SubmitHostLoad(task, w, result.fd, shard);
                if (status.Failure()) { handleFailure(result.fd, status); }
                return;
            }
        }
        AioImpl::Io io;
        io.fd = result.fd;
        io.offset = shard.index * shardSize_;
        io.length = shardSize_;
        io.buffer = shard.addrs.front();
        io.tag = tid;
        io.callback = [this, task, w, fd = result.fd, shardIndex, ino,
                       id](AioImpl::Result ioResult) {
            OnIoCallback<dump>(task, w, fd, shardIndex, ino, id, ioResult);
        };
        auto status = dump ? aio_.WriteAsync(std::move(io)) : aio_.ReadAsync(std::move(io));
        if (status.Failure()) {
            if (status == Status::Timeout()) {
                IncrementAioTimeoutMetric();
            } else {
                IncrementIoErrorMetric();
            }
            handleFailure(result.fd, status);
        }
    }
    template <bool dump>
    void Dispatch(TaskPtr t, WaiterPtr w)
    {
        const auto flags = O_DIRECT | (dump ? (O_CREAT | O_WRONLY) : O_RDONLY);
        const auto number = t->desc.size();
        w->Set(number);
        std::list<BlockOperator::OpenTask> tasks;
        for (size_t i = 0; i < number; ++i) {
            BlockOperator::OpenTask task;
            const auto& shard = t->desc[i];
            task.id = shard.owner;
            task.activated = dump;
            task.flags = flags;
            task.tag = t->id;
            task.callback = [this, t, w, i](BlockOperator::OpenResult result) {
                OnOpenCallback<dump>(t, w, t->desc[i], result);
            };
            tasks.push_back(std::move(task));
        }
        blockOperator_.Submit(std::move(tasks));
    }
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        const auto id = t->id;
        const auto& brief = t->desc.brief;
        const auto num = t->desc.size();
        const auto size = shardSize_ * num;
        const auto tp = w->startTp;
        const auto isDump = (t->type == TransTask::Type::DUMP);
        UC_DEBUG("Posix task({},{},{},{}) dispatching.", id, brief, num, size);
        const auto wait = NowTime::Now() - tp;
        w->SetEpilog([this, id, brief = std::move(brief), num, size, tp, isDump] {
            auto cost = NowTime::Now() - tp;
            auto costMs = cost * 1e3;
            auto bwGbps = cost > 0 ? static_cast<double>(size) / cost / 1e9 : 0.0;
            UC_DEBUG("Posix task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                     costMs);
            static UC::Metrics::CachedMetric loadDuration{"posix_load_task_duration_ms"};
            static UC::Metrics::CachedMetric dumpDuration{"posix_dump_task_duration_ms"};
            static UC::Metrics::CachedMetric loadBandwidth{"posix_s2h_bandwidth_gbps"};
            static UC::Metrics::CachedMetric dumpBandwidth{"posix_h2s_bandwidth_gbps"};
            static UC::Metrics::CachedMetric loadBytes{"posix_s2h_bytes_total"};
            static UC::Metrics::CachedMetric dumpBytes{"posix_h2s_bytes_total"};
            UC::Metrics::UpdateStats(isDump ? dumpDuration : loadDuration, costMs);
            UC::Metrics::UpdateStats(isDump ? dumpBandwidth : loadBandwidth, bwGbps);
            UC::Metrics::UpdateStats(isDump ? dumpBytes : loadBytes, static_cast<double>(size));
            std::lock_guard<std::mutex> lk(regMutex_);
            registry_.erase(id);
        });
        {
            std::lock_guard<std::mutex> lk(regMutex_);
            auto deadline = timeoutMs_ > 0
                                ? NowTime::Now() + (timeoutMs_ + kWatchdogGraceMs) / 1000.0
                                : std::numeric_limits<double>::max();
            registry_[id] = Inflight{t, w, deadline, num, false};
        }
        if (timeoutMs_ > 0) {
            UC_DEBUG("AIO task({}) registered: type={}, shardCount={}, deadlineMs={}.", id,
                     isDump ? "dump" : "load", num, timeoutMs_ + kWatchdogGraceMs);
        } else {
            UC_DEBUG("AIO task({}) registered: type={}, shardCount={}, deadline=disabled.", id,
                     isDump ? "dump" : "load", num);
        }
        if (isDump) {
            UpdateWaitMetrics<true>(wait);
            Dispatch<true>(t, w);
        } else {
            UpdateWaitMetrics<false>(wait);
            Dispatch<false>(t, w);
        }
    }
    void SweepDeadlines()
    {
        auto now = NowTime::Now();
        std::vector<std::tuple<Detail::TaskHandle, WaiterPtr, size_t>> due;
        {
            std::lock_guard<std::mutex> lk(regMutex_);
            for (auto& [id, inf] : registry_) {
                if (!inf.aborted && now >= inf.deadlineTp) {
                    inf.aborted = true;
                    due.emplace_back(id, inf.w, inf.shardCount);
                }
            }
        }
        for (auto& [id, w, shardCount] : due) {
            auto stuckMs = (now - w->startTp) * 1e3;
            auto pending = w->Pending();
            IncrementAioTimeoutMetric();
            UC_ERROR(
                "AIO task({}) timed out after {:.1f}ms; force-draining latch ({} of {} "
                "shard(s) outstanding).",
                id, stuckMs, pending, shardCount);
            ForceComplete(id, w);
        }
    }
    void Cancel(TaskPtr t) override
    {
        WaiterPtr w;
        {
            std::lock_guard<std::mutex> lk(regMutex_);
            auto it = registry_.find(t->id);
            if (it == registry_.end()) { return; }
            if (it->second.aborted) { return; }
            it->second.aborted = true;
            w = it->second.w;
        }
        UC_ERROR("AIO task({}) cancelled on wait timeout; force-draining latch.", t->id);
        IncrementAioTimeoutMetric();
        ForceComplete(t->id, w);
    }
    void ForceComplete(Detail::TaskHandle id, const WaiterPtr& w)
    {
        auto pending = w ? w->Pending() : 0;
        UC_WARN("AIO task({}) force-completing; pending latch count before abort={}.", id, pending);
        failureSet_.Insert(id);
        aio_.CancelTask(id);
        blockOperator_.CancelQueued(id);
        if (w) { w->Abort(); }
    }
};

}  // namespace UC::PosixStore

#endif
