#include "scheduler.hpp"

#include "../../utils.hpp"

#include <algorithm>

namespace llaisys::core::scheduler {

bool SequenceEntry::isFinished() const {
    return status == SequenceStatus::Finished;
}

Scheduler::Scheduler(size_t max_num_seqs,
                     size_t max_num_batched_tokens,
                     int64_t eos_token,
                     std::shared_ptr<paged_kv::BlockManager> block_manager)
    : _max_num_seqs(max_num_seqs),
      _max_num_batched_tokens(max_num_batched_tokens),
      _eos_token(eos_token),
      _block_manager(std::move(block_manager)) {
    CHECK_ARGUMENT(_max_num_seqs > 0, "Scheduler: max_num_seqs must be greater than 0");
    CHECK_ARGUMENT(_max_num_batched_tokens > 0, "Scheduler: max_num_batched_tokens must be greater than 0");
    CHECK_ARGUMENT(_block_manager != nullptr, "Scheduler: block_manager must not be null");
}

bool Scheduler::isFinished() const {
    return _waiting.empty() && _running.empty();
}

std::shared_ptr<SequenceEntry> Scheduler::add(std::shared_ptr<paged_kv::SequenceState> sequence,
                                              size_t max_completion_tokens,
                                              bool ignore_eos) {
    CHECK_ARGUMENT(sequence != nullptr, "Scheduler::add: sequence must not be null");
    auto entry = std::make_shared<SequenceEntry>();
    entry->sequence = std::move(sequence);
    entry->status = SequenceStatus::Waiting;
    entry->max_completion_tokens = max_completion_tokens;
    entry->ignore_eos = ignore_eos;
    _waiting.push_back(entry);
    return entry;
}

ScheduleResult Scheduler::schedule() {
    ScheduleResult result;

    size_t num_seqs = 0;
    size_t num_batched_tokens = 0;
    while (!_waiting.empty() && num_seqs < _max_num_seqs) {
        auto entry = _waiting.front();
        auto &seq = *entry->sequence;
        if (!_block_manager->canAllocate(seq)) {
            break;
        }

        const size_t estimated_cached_tokens = _block_manager->estimateCachedTokens(seq);
        const size_t prefill_tokens = seq.numTokens() - estimated_cached_tokens;
        if (num_batched_tokens + prefill_tokens > _max_num_batched_tokens) {
            break;
        }

        _block_manager->allocate(seq);
        entry->status = SequenceStatus::Running;
        _waiting.pop_front();
        _running.push_back(entry);
        result.scheduled.push_back(entry);
        num_seqs += 1;
        num_batched_tokens += prefill_tokens;
    }

    if (!result.scheduled.empty()) {
        result.is_prefill = true;
        return result;
    }

    while (!_running.empty() && num_seqs < _max_num_seqs) {
        auto entry = _running.front();
        _running.pop_front();
        while (!_block_manager->canAppend(*entry->sequence)) {
            if (!_running.empty()) {
                preempt(_running.back());
                _running.pop_back();
            } else {
                preempt(entry);
                entry.reset();
                break;
            }
        }
        if (!entry) {
            break;
        }
        _block_manager->prepareAppend(*entry->sequence);
        result.scheduled.push_back(entry);
        num_seqs += 1;
    }

    CHECK_ARGUMENT(!result.scheduled.empty(), "Scheduler::schedule: no sequences available to decode");
    for (const auto &entry : result.scheduled) {
        _running.push_back(entry);
    }
    result.is_prefill = false;
    return result;
}

void Scheduler::postprocess(const std::vector<std::shared_ptr<SequenceEntry>> &entries,
                            const std::vector<int64_t> &token_ids) {
    CHECK_ARGUMENT(entries.size() == token_ids.size(), "Scheduler::postprocess: entries/token_ids size mismatch");
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto &entry = entries[i];
        entry->sequence->appendToken(token_ids[i]);
        const bool hit_eos = (!entry->ignore_eos && token_ids[i] == _eos_token);
        const bool hit_max_tokens =
            entry->max_completion_tokens > 0 &&
            entry->sequence->numCompletionTokens() >= entry->max_completion_tokens;
        if (hit_eos || hit_max_tokens) {
            entry->status = SequenceStatus::Finished;
            _block_manager->deallocate(*entry->sequence);
            auto it = std::find(_running.begin(), _running.end(), entry);
            if (it != _running.end()) {
                _running.erase(it);
            }
        }
    }
}

const std::deque<std::shared_ptr<SequenceEntry>> &Scheduler::waiting() const {
    return _waiting;
}

const std::deque<std::shared_ptr<SequenceEntry>> &Scheduler::running() const {
    return _running;
}

void Scheduler::preempt(const std::shared_ptr<SequenceEntry> &entry) {
    entry->status = SequenceStatus::Waiting;
    _block_manager->deallocate(*entry->sequence);
    _waiting.push_front(entry);
}

} // namespace llaisys::core::scheduler
