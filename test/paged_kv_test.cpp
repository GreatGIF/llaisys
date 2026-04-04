#include "../src/core/paged_kv/paged_kv.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using llaisys::core::paged_kv::BlockAllocator;
using llaisys::core::paged_kv::BlockManager;
using llaisys::core::paged_kv::SequenceState;

namespace {

void expect(bool cond, const std::string &message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void expect_invalid_argument_silent(Fn &&fn, const std::string &message) {
    std::ostringstream sink;
    auto *old_buf = std::cerr.rdbuf(sink.rdbuf());
    bool caught = false;
    try {
        fn();
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    std::cerr.rdbuf(old_buf);
    expect(caught, message);
}

void test_allocator_allocate_release() {
    BlockAllocator allocator(4, 16);
    expect(allocator.totalBlocks() == 4, "allocator total block mismatch");
    expect(allocator.freeBlocks() == 4, "allocator free block mismatch");
    expect(allocator.usedBlocks() == 0, "allocator used block mismatch");

    const size_t block0 = allocator.allocate();
    const size_t block1 = allocator.allocate();
    expect(block0 == 0, "first allocated block should be 0");
    expect(block1 == 1, "second allocated block should be 1");
    expect(allocator.freeBlocks() == 2, "free block count after allocate mismatch");
    expect(allocator.usedBlocks() == 2, "used block count after allocate mismatch");

    allocator.release(block0);
    expect(allocator.freeBlocks() == 3, "free block count after release mismatch");
    expect(allocator.usedBlocks() == 1, "used block count after release mismatch");

    const size_t reused = allocator.allocate();
    expect(reused == 2, "allocator should keep fifo free block order");
    expect(allocator.freeBlocks() == 2, "free block count after reuse mismatch");
    expect(allocator.usedBlocks() == 2, "used block count after reuse mismatch");
}

void test_allocator_exhaustion_and_double_release() {
    BlockAllocator allocator(2, 8);
    allocator.allocate();
    allocator.allocate();

    expect_invalid_argument_silent([&]() { allocator.allocate(); },
                                   "allocator exhaustion should throw");

    allocator.release(0);

    expect_invalid_argument_silent([&]() { allocator.release(0); },
                                   "double release should throw");
}

void test_sequence_state_properties() {
    SequenceState seq(7, {11, 12, 13, 14, 15}, 4);
    expect(seq.seqId() == 7, "sequence id mismatch");
    expect(seq.numTokens() == 5, "numTokens mismatch");
    expect(seq.numPromptTokens() == 5, "numPromptTokens mismatch");
    expect(seq.numCompletionTokens() == 0, "numCompletionTokens mismatch");
    expect(seq.numBlocks() == 2, "numBlocks mismatch");
    expect(seq.lastBlockNumTokens() == 1, "lastBlockNumTokens mismatch");
    expect(seq.lastToken() == 15, "lastToken mismatch");

    auto first_block = seq.block(0);
    auto second_block = seq.block(1);
    expect(first_block == std::vector<int64_t>({11, 12, 13, 14}), "first block content mismatch");
    expect(second_block == std::vector<int64_t>({15}), "second block content mismatch");

    seq.appendToken(16);
    seq.appendToken(17);
    expect(seq.numTokens() == 7, "appendToken numTokens mismatch");
    expect(seq.numCompletionTokens() == 2, "appendToken numCompletionTokens mismatch");
    expect(seq.completionTokenIds() == std::vector<int64_t>({16, 17}),
           "completionTokenIds mismatch");
}

void test_block_manager_allocate_deallocate() {
    BlockManager manager(8, 4);
    SequenceState seq(1, {101, 102, 103, 104, 105, 106}, 4);

    expect(manager.canAllocate(seq), "manager should be able to allocate sequence");
    manager.allocate(seq);

    expect(seq.blockTable().size() == 2, "block_table size mismatch after allocate");
    expect(seq.blockTable()[0] == 0, "first block_table entry mismatch");
    expect(seq.blockTable()[1] == 1, "second block_table entry mismatch");
    expect(seq.locate(0).block_id == 0, "locate(0) block id mismatch");
    expect(seq.locate(5).block_id == 1, "locate(5) block id mismatch");
    expect(seq.locate(5).block_offset == 1, "locate(5) block offset mismatch");
    expect(manager.freeBlocks() == 6, "freeBlocks mismatch after allocate");
    expect(manager.usedBlocks() == 2, "usedBlocks mismatch after allocate");

    manager.deallocate(seq);
    expect(seq.blockTable().empty(), "block_table should be empty after deallocate");
    expect(seq.numCachedTokens() == 0, "numCachedTokens should reset after deallocate");
    expect(manager.freeBlocks() == 8, "all blocks should be returned after deallocate");
}

void test_block_manager_prepare_append() {
    BlockManager manager(8, 4);
    SequenceState seq(2, {1, 2, 3, 4}, 4);
    manager.allocate(seq);

    expect(seq.blockTable().size() == 1, "initial block_table size mismatch");
    expect(manager.canAppend(seq), "append at block boundary should not need new block yet");

    seq.appendToken(5);
    expect(seq.numTokens() == 5, "sequence length after append mismatch");
    expect(manager.canAppend(seq), "sequence should be appendable with one extra free block");
    manager.prepareAppend(seq);
    expect(seq.blockTable().size() == 2, "prepareAppend should extend block_table");
    expect(seq.blockTable()[1] == 1, "newly appended block id mismatch");

    seq.appendToken(6);
    manager.prepareAppend(seq);
    expect(seq.blockTable().size() == 2, "prepareAppend should not add block inside partially filled block");

    seq.appendToken(7);
    seq.appendToken(8);
    manager.prepareAppend(seq);
    expect(seq.blockTable().size() == 2, "prepareAppend should not add block at full block completion");
}

void test_block_manager_capacity_guard() {
    BlockManager manager(2, 4);
    SequenceState seq0(0, {1, 2, 3, 4}, 4);
    SequenceState seq1(1, {5, 6, 7, 8}, 4);
    SequenceState seq2(2, {9, 10, 11, 12}, 4);

    manager.allocate(seq0);
    manager.allocate(seq1);
    expect(!manager.canAllocate(seq2), "manager should reject allocate when blocks are exhausted");

    seq0.appendToken(100);
    expect(!manager.canAppend(seq0), "manager should reject append when no free block remains");
}

void test_prefix_cache_reuse_after_deallocate() {
    BlockManager manager(4, 4);
    SequenceState seq0(0, {10, 11, 12, 13, 20, 21}, 4);
    manager.allocate(seq0);
    expect(seq0.numCachedTokens() == 0, "first allocation should not report cached tokens");
    expect(manager.usedBlocks() == 2, "first allocation should consume two blocks");
    manager.deallocate(seq0);
    expect(manager.freeBlocks() == 4, "all blocks should be free after deallocate");

    SequenceState seq1(1, {10, 11, 12, 13, 20, 21}, 4);
    expect(manager.canAllocate(seq1), "prefix cache reuse should make re-allocation possible");
    manager.allocate(seq1);
    expect(seq1.numCachedTokens() == 4, "second allocation should reuse one cached full block");
    expect(seq1.blockTable().size() == 2, "reused sequence block_table size mismatch");
    expect(seq1.blockTable()[0] == 0, "reused first block id mismatch");
    expect(manager.usedBlocks() == 2, "reuse plus partial miss should use two unique blocks");
}

void test_prefix_cache_shared_refcount() {
    BlockManager manager(4, 4);
    SequenceState seq0(0, {1, 2, 3, 4}, 4);
    manager.allocate(seq0);
    expect(manager.usedBlocks() == 1, "single full block sequence should use one block");

    SequenceState seq1(1, {1, 2, 3, 4}, 4);
    expect(manager.canAllocate(seq1), "identical prefix should be allocatable by sharing cache");
    manager.allocate(seq1);
    expect(seq1.numCachedTokens() == 4, "shared full block should count as cached");
    expect(seq0.blockTable()[0] == seq1.blockTable()[0], "identical sequences should share block id");
    expect(manager.usedBlocks() == 1, "shared prefix should not consume an extra block");

    manager.deallocate(seq0);
    expect(manager.usedBlocks() == 1, "shared block should stay allocated while second sequence uses it");
    expect(seq1.locate(3).block_id == seq1.blockTable()[0], "remaining sequence should keep valid mapping");

    manager.deallocate(seq1);
    expect(manager.freeBlocks() == 4, "shared block should be released after last owner exits");
}

void test_prefix_cache_finalize_on_append() {
    BlockManager manager(4, 4);
    SequenceState seq0(0, {1, 2, 3, 4}, 4);
    manager.allocate(seq0);

    seq0.appendToken(5);
    manager.prepareAppend(seq0);
    seq0.appendToken(6);
    manager.prepareAppend(seq0);
    seq0.appendToken(7);
    manager.prepareAppend(seq0);
    seq0.appendToken(8);
    manager.prepareAppend(seq0);
    manager.deallocate(seq0);

    SequenceState seq1(1, {1, 2, 3, 4, 5, 6, 7, 8}, 4);
    manager.allocate(seq1);
    expect(seq1.numCachedTokens() == 8, "finalized full appended block should also be reusable");
    expect(seq1.blockTable().size() == 2, "two-block reused sequence should have two block ids");
}

void test_prepare_prefill_without_prefix_cache() {
    BlockManager manager(8, 4);
    SequenceState seq0(0, {10, 11, 12}, 4);
    SequenceState seq1(1, {20, 21, 22, 23, 24}, 4);
    manager.allocate(seq0);
    manager.allocate(seq1);

    auto batch = BlockManager::preparePrefill({&seq0, &seq1}, 4);
    expect(batch.input_ids == std::vector<int64_t>({10, 11, 12, 20, 21, 22, 23, 24}),
           "prefill input_ids mismatch without prefix cache");
    expect(batch.positions == std::vector<int64_t>({0, 1, 2, 0, 1, 2, 3, 4}),
           "prefill positions mismatch without prefix cache");
    expect(batch.cu_seqlens_q == std::vector<int32_t>({0, 3, 8}),
           "prefill cu_seqlens_q mismatch without prefix cache");
    expect(batch.cu_seqlens_k == std::vector<int32_t>({0, 3, 8}),
           "prefill cu_seqlens_k mismatch without prefix cache");
    expect(batch.max_seqlen_q == 5, "prefill max_seqlen_q mismatch without prefix cache");
    expect(batch.max_seqlen_k == 5, "prefill max_seqlen_k mismatch without prefix cache");
    expect(batch.slot_mapping == std::vector<int32_t>({0, 1, 2, 4, 5, 6, 7, 8}),
           "prefill slot_mapping mismatch without prefix cache");
    expect(batch.block_tables.empty(), "prefill block_tables should be empty without prefix cache");
}

void test_prepare_prefill_with_prefix_cache() {
    BlockManager manager(8, 4);
    SequenceState warmup(0, {1, 2, 3, 4, 5, 6}, 4);
    manager.allocate(warmup);
    manager.deallocate(warmup);

    SequenceState seq0(1, {1, 2, 3, 4, 9, 10}, 4);
    SequenceState seq1(2, {30, 31, 32}, 4);
    manager.allocate(seq0);
    manager.allocate(seq1);

    expect(seq0.numCachedTokens() == 4, "seq0 should reuse one cached full block");
    expect(seq1.numCachedTokens() == 0, "seq1 should have no cached tokens");

    auto batch = BlockManager::preparePrefill({&seq0, &seq1}, 4);
    expect(batch.input_ids == std::vector<int64_t>({9, 10, 30, 31, 32}),
           "prefill input_ids mismatch with prefix cache");
    expect(batch.positions == std::vector<int64_t>({4, 5, 0, 1, 2}),
           "prefill positions mismatch with prefix cache");
    expect(batch.cu_seqlens_q == std::vector<int32_t>({0, 2, 5}),
           "prefill cu_seqlens_q mismatch with prefix cache");
    expect(batch.cu_seqlens_k == std::vector<int32_t>({0, 6, 9}),
           "prefill cu_seqlens_k mismatch with prefix cache");
    expect(batch.max_seqlen_q == 3, "prefill max_seqlen_q mismatch with prefix cache");
    expect(batch.max_seqlen_k == 6, "prefill max_seqlen_k mismatch with prefix cache");
    std::vector<int32_t> expected_slot_mapping = {
        static_cast<int32_t>(seq0.blockTable()[1] * 4 + 0),
        static_cast<int32_t>(seq0.blockTable()[1] * 4 + 1),
        static_cast<int32_t>(seq1.blockTable()[0] * 4 + 0),
        static_cast<int32_t>(seq1.blockTable()[0] * 4 + 1),
        static_cast<int32_t>(seq1.blockTable()[0] * 4 + 2),
    };
    expect(batch.slot_mapping == expected_slot_mapping,
           "prefill slot_mapping mismatch with prefix cache");
    expect(batch.block_tables.size() == 2, "prefill block_tables batch size mismatch with prefix cache");
    expect(batch.block_tables[0] == std::vector<int32_t>({
               static_cast<int32_t>(seq0.blockTable()[0]),
               static_cast<int32_t>(seq0.blockTable()[1]),
           }),
           "prefill padded block_table mismatch for seq0");
    expect(batch.block_tables[1] == std::vector<int32_t>({
               static_cast<int32_t>(seq1.blockTable()[0]),
               -1,
           }),
           "prefill padded block_table mismatch for seq1");
}

void test_prepare_decode() {
    BlockManager manager(8, 4);
    SequenceState seq0(0, {10, 11, 12, 13, 14}, 4);
    SequenceState seq1(1, {20, 21, 22}, 4);
    manager.allocate(seq0);
    manager.allocate(seq1);

    auto batch = BlockManager::prepareDecode({&seq0, &seq1}, 4);
    expect(batch.input_ids == std::vector<int64_t>({14, 22}),
           "decode input_ids mismatch");
    expect(batch.positions == std::vector<int64_t>({4, 2}),
           "decode positions mismatch");
    expect(batch.context_lens == std::vector<int32_t>({5, 3}),
           "decode context_lens mismatch");
    expect(batch.slot_mapping == std::vector<int32_t>({4, 10}),
           "decode slot_mapping mismatch");
    expect(batch.block_tables.size() == 2, "decode block_tables batch size mismatch");
    expect(batch.block_tables[0] == std::vector<int32_t>({0, 1}),
           "decode padded block_table mismatch for seq0");
    expect(batch.block_tables[1] == std::vector<int32_t>({2, -1}),
           "decode padded block_table mismatch for seq1");
}

} // namespace

int main() {
    test_allocator_allocate_release();
    test_allocator_exhaustion_and_double_release();
    test_sequence_state_properties();
    test_block_manager_allocate_deallocate();
    test_block_manager_prepare_append();
    test_block_manager_capacity_guard();
    test_prefix_cache_reuse_after_deallocate();
    test_prefix_cache_shared_refcount();
    test_prefix_cache_finalize_on_append();
    test_prepare_prefill_without_prefix_cache();
    test_prepare_prefill_with_prefix_cache();
    test_prepare_decode();
    std::cout << "paged_kv_test passed" << std::endl;
    return 0;
}
