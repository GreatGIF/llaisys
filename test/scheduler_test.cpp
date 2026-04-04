#include "../src/core/paged_kv/paged_kv.hpp"
#include "../src/core/scheduler/scheduler.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using llaisys::core::paged_kv::BlockManager;
using llaisys::core::paged_kv::SequenceState;
using llaisys::core::scheduler::ScheduleResult;
using llaisys::core::scheduler::Scheduler;
using llaisys::core::scheduler::SequenceStatus;

namespace {

void expect(bool cond, const std::string &message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

std::shared_ptr<SequenceState> make_seq(size_t seq_id, std::initializer_list<int64_t> token_ids, size_t block_size = 4) {
    return std::make_shared<SequenceState>(seq_id, std::vector<int64_t>(token_ids), block_size);
}

void test_prefill_first_returns_immediately() {
    auto block_manager = std::make_shared<BlockManager>(16, 4);
    Scheduler scheduler(4, 64, 99, block_manager);

    auto running = scheduler.add(make_seq(0, {1, 2, 3}), 8);
    auto first = scheduler.schedule();
    expect(first.is_prefill, "first schedule should be prefill");
    expect(first.scheduled.size() == 1, "first schedule should include one sequence");

    auto waiting = scheduler.add(make_seq(1, {4, 5}), 8);
    auto second = scheduler.schedule();
    expect(second.is_prefill, "prefill queue should take priority over decode");
    expect(second.scheduled.size() == 1, "second schedule should return immediately after prefill");
    expect(second.scheduled[0] == waiting, "second schedule should pick the waiting prefill sequence");
    expect(running->status == SequenceStatus::Running, "existing running sequence should remain running");
}

void test_decode_only_when_prefill_empty() {
    auto block_manager = std::make_shared<BlockManager>(16, 4);
    Scheduler scheduler(4, 64, 99, block_manager);

    auto entry0 = scheduler.add(make_seq(0, {1, 2, 3}), 8);
    auto entry1 = scheduler.add(make_seq(1, {4, 5}), 8);
    auto prefill = scheduler.schedule();
    expect(prefill.is_prefill, "first schedule should be prefill");
    expect(prefill.scheduled.size() == 2, "both waiting sequences should join same prefill batch");

    auto decode = scheduler.schedule();
    expect(!decode.is_prefill, "decode should run only after waiting queue is empty");
    expect(decode.scheduled.size() == 2, "decode should schedule running sequences");
    expect(decode.scheduled[0] == entry0, "decode should preserve running order for first sequence");
    expect(decode.scheduled[1] == entry1, "decode should preserve running order for second sequence");
}

void test_prefill_respects_max_batched_tokens() {
    auto block_manager = std::make_shared<BlockManager>(16, 4);
    Scheduler scheduler(4, 5, 99, block_manager);

    auto entry0 = scheduler.add(make_seq(0, {1, 2, 3}), 8);
    auto entry1 = scheduler.add(make_seq(1, {4, 5, 6}), 8);
    auto result = scheduler.schedule();
    expect(result.is_prefill, "schedule should still be prefill");
    expect(result.scheduled.size() == 1, "prefill batch should stop at max_num_batched_tokens");
    expect(result.scheduled[0] == entry0, "only first waiting sequence should be scheduled");
    expect(scheduler.waiting().size() == 1, "second sequence should remain waiting");
    expect(scheduler.waiting().front() == entry1, "second sequence should still be at waiting front");
}

void test_prefill_uses_prefix_cache_to_fit_budget() {
    auto block_manager = std::make_shared<BlockManager>(16, 4);
    Scheduler warmup_scheduler(4, 4, 99, block_manager);

    auto warmup = warmup_scheduler.add(make_seq(0, {1, 2, 3, 4}), 1);
    auto first = warmup_scheduler.schedule();
    expect(first.is_prefill, "warmup should prefill");
    warmup_scheduler.postprocess(first.scheduled, {50});
    expect(warmup->isFinished(), "warmup sequence should finish after one completion token");

    Scheduler scheduler(4, 2, 99, block_manager);
    auto reused = scheduler.add(make_seq(1, {1, 2, 3, 4, 5, 6}), 8);
    expect(block_manager->estimateCachedTokens(*reused->sequence) == 4,
           "reused sequence should estimate four cached tokens");
    expect(block_manager->canAllocate(*reused->sequence),
           "reused sequence should be allocatable with cached prefix");
    auto result = scheduler.schedule();
    expect(result.is_prefill, "reused prompt should still prefill");
    expect(result.scheduled.size() == 1, "reused prompt should fit alone");
    expect(result.scheduled[0] == reused, "reused sequence should be scheduled");
    expect(reused->sequence->numCachedTokens() == 4, "reused prefix should reduce real prefill token count");
}

void test_decode_preempts_when_append_cannot_fit() {
    auto block_manager = std::make_shared<BlockManager>(2, 4);
    Scheduler scheduler(4, 64, 99, block_manager);

    auto entry0 = scheduler.add(make_seq(0, {1, 2, 3, 4}), 8);
    auto entry1 = scheduler.add(make_seq(1, {5, 6, 7, 8}), 8);
    auto prefill = scheduler.schedule();
    expect(prefill.scheduled.size() == 2, "both sequences should prefill into the two available blocks");

    scheduler.postprocess({entry0}, {10});
    auto decode = scheduler.schedule();
    expect(!decode.is_prefill, "with no waiting seqs, scheduler should try decode");
    expect(decode.scheduled.size() == 1, "scheduler should preempt one sequence to free space for decode");
    expect(decode.scheduled[0] == entry0, "sequence needing append should be scheduled");
    expect(entry1->status == SequenceStatus::Waiting, "other running sequence should be preempted back to waiting");
    expect(entry1->sequence->blockTable().empty(), "preempted sequence should release its block table");
}

} // namespace

int main() {
    test_prefill_first_returns_immediately();
    test_decode_only_when_prefill_empty();
    test_prefill_respects_max_batched_tokens();
    test_prefill_uses_prefix_cache_to_fit_budget();
    test_decode_preempts_when_append_cannot_fit();
    std::cout << "scheduler_test passed" << std::endl;
    return 0;
}
