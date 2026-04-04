#pragma once

#include "../paged_kv/paged_kv.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace llaisys::core::scheduler {

enum class SequenceStatus {
    Waiting,
    Running,
    Finished,
};

struct SequenceEntry {
    std::shared_ptr<paged_kv::SequenceState> sequence;
    SequenceStatus status = SequenceStatus::Waiting;
    size_t max_completion_tokens = 0;
    bool ignore_eos = false;

    bool isFinished() const;
};

struct ScheduleResult {
    std::vector<std::shared_ptr<SequenceEntry>> scheduled;
    bool is_prefill = false;
};

class Scheduler {
public:
    Scheduler(size_t max_num_seqs,
              size_t max_num_batched_tokens,
              int64_t eos_token,
              std::shared_ptr<paged_kv::BlockManager> block_manager);

    bool isFinished() const;

    std::shared_ptr<SequenceEntry> add(std::shared_ptr<paged_kv::SequenceState> sequence,
                                       size_t max_completion_tokens,
                                       bool ignore_eos = false);

    ScheduleResult schedule();
    void postprocess(const std::vector<std::shared_ptr<SequenceEntry>> &entries,
                     const std::vector<int64_t> &token_ids);

    const std::deque<std::shared_ptr<SequenceEntry>> &waiting() const;
    const std::deque<std::shared_ptr<SequenceEntry>> &running() const;

private:
    void preempt(const std::shared_ptr<SequenceEntry> &entry);

    size_t _max_num_seqs;
    size_t _max_num_batched_tokens;
    int64_t _eos_token;
    std::shared_ptr<paged_kv::BlockManager> _block_manager;
    std::deque<std::shared_ptr<SequenceEntry>> _waiting;
    std::deque<std::shared_ptr<SequenceEntry>> _running;
};

} // namespace llaisys::core::scheduler
