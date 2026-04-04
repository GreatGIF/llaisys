#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

namespace llaisys::core::paged_kv {

struct BlockLocation {
    size_t block_id;
    size_t block_offset;
};

struct PrefillBatch {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> positions;
    std::vector<int32_t> cu_seqlens_q;
    std::vector<int32_t> cu_seqlens_k;
    size_t max_seqlen_q = 0;
    size_t max_seqlen_k = 0;
    std::vector<int32_t> slot_mapping;
    std::vector<std::vector<int32_t>> block_tables;
};

struct DecodeBatch {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> positions;
    std::vector<int32_t> slot_mapping;
    std::vector<int32_t> context_lens;
    std::vector<std::vector<int32_t>> block_tables;
};

class BlockAllocator {
public:
    BlockAllocator(size_t total_blocks, size_t block_size);

    size_t blockSize() const;
    size_t totalBlocks() const;
    size_t freeBlocks() const;
    size_t usedBlocks() const;

    size_t allocate();
    std::vector<size_t> allocate(size_t count);
    void acquire(size_t block_id);

    void release(size_t block_id);
    void release(const std::vector<size_t> &block_ids);

private:
    size_t _total_blocks;
    size_t _block_size;
    std::deque<size_t> _free_blocks;
    std::vector<bool> _allocated;
};

class SequenceState {
public:
    SequenceState(size_t seq_id, std::vector<int64_t> token_ids, size_t block_size);

    size_t seqId() const;
    size_t blockSize() const;
    size_t numTokens() const;
    size_t numPromptTokens() const;
    size_t numCompletionTokens() const;
    size_t numCachedTokens() const;
    size_t numCachedBlocks() const;
    size_t numBlocks() const;
    size_t lastBlockNumTokens() const;
    int64_t lastToken() const;

    const std::vector<int64_t> &tokenIds() const;
    std::vector<int64_t> promptTokenIds() const;
    std::vector<int64_t> completionTokenIds() const;
    std::vector<int64_t> block(size_t block_index) const;

    const std::vector<size_t> &blockTable() const;
    std::vector<size_t> &mutableBlockTable();

    bool ownsToken(size_t token_index) const;
    BlockLocation locate(size_t token_index) const;

    void appendToken(int64_t token_id);
    void setNumCachedTokens(size_t num_cached_tokens);
    void clearPagedState();

private:
    size_t _seq_id;
    size_t _block_size;
    std::vector<int64_t> _token_ids;
    size_t _num_prompt_tokens;
    size_t _num_cached_tokens;
    std::vector<size_t> _block_table;
};

class BlockManager {
public:
    BlockManager(size_t total_blocks, size_t block_size);

    size_t blockSize() const;
    size_t totalBlocks() const;
    size_t freeBlocks() const;
    size_t usedBlocks() const;

    size_t estimateCachedTokens(const SequenceState &seq) const;
    bool canAllocate(const SequenceState &seq) const;
    bool canAppend(const SequenceState &seq) const;

    void allocate(SequenceState &seq);
    void deallocate(SequenceState &seq);
    void prepareAppend(SequenceState &seq);

    static PrefillBatch preparePrefill(const std::vector<SequenceState *> &seqs, size_t block_size);
    static DecodeBatch prepareDecode(const std::vector<SequenceState *> &seqs, size_t block_size);

private:
    struct BlockEntry {
        size_t ref_count = 0;
        std::optional<uint64_t> hash;
        std::vector<int64_t> token_ids;
    };

    static uint64_t computeHash(const std::vector<int64_t> &token_ids, std::optional<uint64_t> prefix_hash);

    void retain(size_t block_id);
    void release(size_t block_id);
    void invalidateCache(size_t block_id);
    void updateCache(size_t block_id, uint64_t hash, const std::vector<int64_t> &token_ids);
    bool canReuse(uint64_t hash, const std::vector<int64_t> &token_ids) const;
    size_t cachedBlockId(uint64_t hash) const;
    void finalizeLastFullBlock(SequenceState &seq);

    BlockAllocator _allocator;
    std::vector<BlockEntry> _blocks;
    std::unordered_map<uint64_t, size_t> _hash_to_block_id;
};

} // namespace llaisys::core::paged_kv
