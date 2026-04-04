#include "paged_kv.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <functional>

namespace llaisys::core::paged_kv {

BlockAllocator::BlockAllocator(size_t total_blocks, size_t block_size)
    : _total_blocks(total_blocks), _block_size(block_size), _allocated(total_blocks, false) {
    CHECK_ARGUMENT(total_blocks > 0, "BlockAllocator: total_blocks must be greater than 0");
    CHECK_ARGUMENT(block_size > 0, "BlockAllocator: block_size must be greater than 0");

    for (size_t block_id = 0; block_id < total_blocks; ++block_id) {
        _free_blocks.push_back(block_id);
    }
}

size_t BlockAllocator::blockSize() const {
    return _block_size;
}

size_t BlockAllocator::totalBlocks() const {
    return _total_blocks;
}

size_t BlockAllocator::freeBlocks() const {
    return _free_blocks.size();
}

size_t BlockAllocator::usedBlocks() const {
    return _total_blocks - _free_blocks.size();
}

size_t BlockAllocator::allocate() {
    CHECK_ARGUMENT(!_free_blocks.empty(), "BlockAllocator: out of free blocks");
    const size_t block_id = _free_blocks.front();
    _free_blocks.pop_front();
    _allocated[block_id] = true;
    return block_id;
}

std::vector<size_t> BlockAllocator::allocate(size_t count) {
    CHECK_ARGUMENT(count <= freeBlocks(), "BlockAllocator: insufficient free blocks");

    std::vector<size_t> block_ids;
    block_ids.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        block_ids.push_back(allocate());
    }
    return block_ids;
}

void BlockAllocator::acquire(size_t block_id) {
    CHECK_ARGUMENT(block_id < _total_blocks, "BlockAllocator: block_id out of range");
    CHECK_ARGUMENT(!_allocated[block_id], "BlockAllocator: block already allocated");
    auto it = std::find(_free_blocks.begin(), _free_blocks.end(), block_id);
    CHECK_ARGUMENT(it != _free_blocks.end(), "BlockAllocator: block is not present in free list");
    _free_blocks.erase(it);
    _allocated[block_id] = true;
}

void BlockAllocator::release(size_t block_id) {
    CHECK_ARGUMENT(block_id < _total_blocks, "BlockAllocator: block_id out of range");
    CHECK_ARGUMENT(_allocated[block_id], "BlockAllocator: block already released");
    _allocated[block_id] = false;
    _free_blocks.push_back(block_id);
}

void BlockAllocator::release(const std::vector<size_t> &block_ids) {
    for (size_t block_id : block_ids) {
        release(block_id);
    }
}

SequenceState::SequenceState(size_t seq_id, std::vector<int64_t> token_ids, size_t block_size)
    : _seq_id(seq_id),
      _block_size(block_size),
      _token_ids(std::move(token_ids)),
      _num_prompt_tokens(_token_ids.size()),
      _num_cached_tokens(0) {
    CHECK_ARGUMENT(block_size > 0, "SequenceState: block_size must be greater than 0");
    CHECK_ARGUMENT(!_token_ids.empty(), "SequenceState: token_ids must not be empty");
}

size_t SequenceState::seqId() const {
    return _seq_id;
}

size_t SequenceState::blockSize() const {
    return _block_size;
}

size_t SequenceState::numTokens() const {
    return _token_ids.size();
}

size_t SequenceState::numPromptTokens() const {
    return _num_prompt_tokens;
}

size_t SequenceState::numCompletionTokens() const {
    return numTokens() - _num_prompt_tokens;
}

size_t SequenceState::numCachedTokens() const {
    return _num_cached_tokens;
}

size_t SequenceState::numCachedBlocks() const {
    return _num_cached_tokens / _block_size;
}

size_t SequenceState::numBlocks() const {
    return (numTokens() + _block_size - 1) / _block_size;
}

size_t SequenceState::lastBlockNumTokens() const {
    return numTokens() - (numBlocks() - 1) * _block_size;
}

int64_t SequenceState::lastToken() const {
    return _token_ids.back();
}

const std::vector<int64_t> &SequenceState::tokenIds() const {
    return _token_ids;
}

std::vector<int64_t> SequenceState::promptTokenIds() const {
    return std::vector<int64_t>(_token_ids.begin(), _token_ids.begin() + static_cast<ptrdiff_t>(_num_prompt_tokens));
}

std::vector<int64_t> SequenceState::completionTokenIds() const {
    return std::vector<int64_t>(_token_ids.begin() + static_cast<ptrdiff_t>(_num_prompt_tokens), _token_ids.end());
}

std::vector<int64_t> SequenceState::block(size_t block_index) const {
    CHECK_ARGUMENT(block_index < numBlocks(), "SequenceState: block_index out of range");
    const size_t start = block_index * _block_size;
    const size_t end = std::min(start + _block_size, numTokens());
    return std::vector<int64_t>(_token_ids.begin() + static_cast<ptrdiff_t>(start),
                                _token_ids.begin() + static_cast<ptrdiff_t>(end));
}

const std::vector<size_t> &SequenceState::blockTable() const {
    return _block_table;
}

std::vector<size_t> &SequenceState::mutableBlockTable() {
    return _block_table;
}

bool SequenceState::ownsToken(size_t token_index) const {
    return token_index < numTokens();
}

BlockLocation SequenceState::locate(size_t token_index) const {
    CHECK_ARGUMENT(token_index < numTokens(), "SequenceState: token_index out of range");
    CHECK_ARGUMENT(_block_table.size() == numBlocks(),
                   "SequenceState: block_table must be allocated before locate");
    const size_t block_index = token_index / _block_size;
    return BlockLocation{_block_table.at(block_index), token_index % _block_size};
}

void SequenceState::appendToken(int64_t token_id) {
    _token_ids.push_back(token_id);
}

void SequenceState::setNumCachedTokens(size_t num_cached_tokens) {
    CHECK_ARGUMENT(num_cached_tokens <= numTokens(),
                   "SequenceState: num_cached_tokens exceeds num_tokens");
    _num_cached_tokens = num_cached_tokens;
}

void SequenceState::clearPagedState() {
    _num_cached_tokens = 0;
    _block_table.clear();
}

BlockManager::BlockManager(size_t total_blocks, size_t block_size)
    : _allocator(total_blocks, block_size), _blocks(total_blocks) {
}

size_t BlockManager::blockSize() const {
    return _allocator.blockSize();
}

size_t BlockManager::totalBlocks() const {
    return _allocator.totalBlocks();
}

size_t BlockManager::freeBlocks() const {
    return _allocator.freeBlocks();
}

size_t BlockManager::usedBlocks() const {
    return _allocator.usedBlocks();
}

size_t BlockManager::estimateCachedTokens(const SequenceState &seq) const {
    if (seq.blockSize() != blockSize()) {
        return 0;
    }
    bool cache_miss = false;
    size_t cached_tokens = 0;
    std::optional<uint64_t> prefix_hash;
    for (size_t block_index = 0; block_index < seq.numBlocks(); ++block_index) {
        const auto token_ids = seq.block(block_index);
        const bool is_full_block = token_ids.size() == blockSize();
        std::optional<uint64_t> block_hash;
        if (is_full_block) {
            block_hash = computeHash(token_ids, prefix_hash);
        }
        if (!cache_miss && block_hash.has_value() && canReuse(*block_hash, token_ids)) {
            cached_tokens += blockSize();
            prefix_hash = block_hash;
            continue;
        }
        cache_miss = true;
        prefix_hash = block_hash;
    }
    return cached_tokens;
}

bool BlockManager::canAllocate(const SequenceState &seq) const {
    if (seq.blockSize() != blockSize()) {
        return false;
    }
    size_t needed_blocks = 0;
    bool cache_miss = false;
    std::optional<uint64_t> prefix_hash;
    for (size_t block_index = 0; block_index < seq.numBlocks(); ++block_index) {
        const auto token_ids = seq.block(block_index);
        const bool is_full_block = token_ids.size() == blockSize();
        std::optional<uint64_t> block_hash;
        if (is_full_block) {
            block_hash = computeHash(token_ids, prefix_hash);
        }
        if (!cache_miss && block_hash.has_value() && canReuse(*block_hash, token_ids)) {
            const size_t block_id = cachedBlockId(*block_hash);
            if (_blocks[block_id].ref_count == 0) {
                needed_blocks += 1;
            }
            prefix_hash = block_hash;
            continue;
        }
        cache_miss = true;
        needed_blocks += 1;
        prefix_hash = block_hash;
    }
    return freeBlocks() >= needed_blocks;
}

bool BlockManager::canAppend(const SequenceState &seq) const {
    CHECK_ARGUMENT(seq.blockSize() == blockSize(), "BlockManager: block_size mismatch");
    CHECK_ARGUMENT(!seq.blockTable().empty(), "BlockManager: sequence must be allocated before append");
    const bool needs_new_block = (seq.numTokens() % blockSize()) == 1;
    return freeBlocks() >= static_cast<size_t>(needs_new_block);
}

void BlockManager::allocate(SequenceState &seq) {
    CHECK_ARGUMENT(seq.blockSize() == blockSize(), "BlockManager: block_size mismatch");
    CHECK_ARGUMENT(seq.blockTable().empty(), "BlockManager: sequence already allocated");
    CHECK_ARGUMENT(canAllocate(seq), "BlockManager: insufficient free blocks for allocate");

    std::vector<size_t> block_table;
    block_table.reserve(seq.numBlocks());
    size_t num_cached_tokens = 0;
    bool cache_miss = false;
    std::optional<uint64_t> prefix_hash;

    for (size_t block_index = 0; block_index < seq.numBlocks(); ++block_index) {
        const auto token_ids = seq.block(block_index);
        const bool is_full_block = token_ids.size() == blockSize();
        std::optional<uint64_t> block_hash;
        if (is_full_block) {
            block_hash = computeHash(token_ids, prefix_hash);
        }

        size_t block_id = 0;
        bool reused = false;
        if (!cache_miss && block_hash.has_value() && canReuse(*block_hash, token_ids)) {
            block_id = cachedBlockId(*block_hash);
            if (_blocks[block_id].ref_count == 0) {
                _allocator.acquire(block_id);
            }
            retain(block_id);
            num_cached_tokens += blockSize();
            reused = true;
        } else {
            cache_miss = true;
            block_id = _allocator.allocate();
            retain(block_id);
            invalidateCache(block_id);
            if (block_hash.has_value()) {
                updateCache(block_id, *block_hash, token_ids);
            } else {
                _blocks[block_id].token_ids = token_ids;
            }
        }
        block_table.push_back(block_id);
        prefix_hash = reused || block_hash.has_value() ? block_hash : std::nullopt;
    }

    seq.mutableBlockTable() = std::move(block_table);
    seq.setNumCachedTokens(num_cached_tokens);
}

void BlockManager::deallocate(SequenceState &seq) {
    CHECK_ARGUMENT(seq.blockSize() == blockSize(), "BlockManager: block_size mismatch");
    for (size_t block_id : seq.blockTable()) {
        release(block_id);
    }
    seq.clearPagedState();
}

void BlockManager::prepareAppend(SequenceState &seq) {
    CHECK_ARGUMENT(seq.blockSize() == blockSize(), "BlockManager: block_size mismatch");
    CHECK_ARGUMENT(!seq.blockTable().empty(), "BlockManager: sequence must be allocated before append");

    if ((seq.numTokens() % blockSize()) == 1) {
        CHECK_ARGUMENT(freeBlocks() > 0, "BlockManager: insufficient free blocks for append");
        const size_t block_id = _allocator.allocate();
        retain(block_id);
        seq.mutableBlockTable().push_back(block_id);
        _blocks[block_id].token_ids.clear();
    } else if ((seq.numTokens() % blockSize()) == 0) {
        finalizeLastFullBlock(seq);
    }
}

PrefillBatch BlockManager::preparePrefill(const std::vector<SequenceState *> &seqs, size_t block_size) {
    CHECK_ARGUMENT(block_size > 0, "BlockManager::preparePrefill: block_size must be greater than 0");
    CHECK_ARGUMENT(!seqs.empty(), "BlockManager::preparePrefill: seqs must not be empty");

    PrefillBatch batch;
    batch.cu_seqlens_q.push_back(0);
    batch.cu_seqlens_k.push_back(0);

    size_t max_block_count = 0;
    bool has_prefix_cache = false;

    for (const SequenceState *seq : seqs) {
        CHECK_ARGUMENT(seq != nullptr, "BlockManager::preparePrefill: seq pointer must not be null");
        CHECK_ARGUMENT(seq->blockSize() == block_size,
                       "BlockManager::preparePrefill: block_size mismatch");
        CHECK_ARGUMENT(seq->blockTable().size() == seq->numBlocks(),
                       "BlockManager::preparePrefill: sequence must have allocated block_table");
        CHECK_ARGUMENT(seq->numCachedTokens() <= seq->numTokens(),
                       "BlockManager::preparePrefill: num_cached_tokens exceeds num_tokens");

        const size_t seqlen = seq->numTokens();
        const size_t seqlen_q = seqlen - seq->numCachedTokens();
        const size_t seqlen_k = seqlen;

        for (size_t token_index = seq->numCachedTokens(); token_index < seqlen; ++token_index) {
            batch.input_ids.push_back(seq->tokenIds()[token_index]);
            batch.positions.push_back(static_cast<int64_t>(token_index));
        }

        batch.cu_seqlens_q.push_back(batch.cu_seqlens_q.back() + static_cast<int32_t>(seqlen_q));
        batch.cu_seqlens_k.push_back(batch.cu_seqlens_k.back() + static_cast<int32_t>(seqlen_k));
        batch.max_seqlen_q = std::max(batch.max_seqlen_q, seqlen_q);
        batch.max_seqlen_k = std::max(batch.max_seqlen_k, seqlen_k);
        max_block_count = std::max(max_block_count, seq->blockTable().size());

        if (seq->numCachedTokens() > 0) {
            has_prefix_cache = true;
        }

        for (size_t block_index = seq->numCachedBlocks(); block_index < seq->numBlocks(); ++block_index) {
            const size_t block_id = seq->blockTable()[block_index];
            const size_t start = block_id * block_size;
            const size_t token_count = (block_index == seq->numBlocks() - 1) ? seq->lastBlockNumTokens() : block_size;
            for (size_t offset = 0; offset < token_count; ++offset) {
                batch.slot_mapping.push_back(static_cast<int32_t>(start + offset));
            }
        }
    }

    CHECK_ARGUMENT(batch.slot_mapping.size() == batch.input_ids.size(),
                   "BlockManager::preparePrefill: slot_mapping size mismatch");

    if (has_prefix_cache) {
        batch.block_tables.reserve(seqs.size());
        for (const SequenceState *seq : seqs) {
            std::vector<int32_t> padded(max_block_count, -1);
            for (size_t i = 0; i < seq->blockTable().size(); ++i) {
                padded[i] = static_cast<int32_t>(seq->blockTable()[i]);
            }
            batch.block_tables.push_back(std::move(padded));
        }
    }

    return batch;
}

DecodeBatch BlockManager::prepareDecode(const std::vector<SequenceState *> &seqs, size_t block_size) {
    CHECK_ARGUMENT(block_size > 0, "BlockManager::prepareDecode: block_size must be greater than 0");
    CHECK_ARGUMENT(!seqs.empty(), "BlockManager::prepareDecode: seqs must not be empty");

    DecodeBatch batch;
    size_t max_block_count = 0;

    for (const SequenceState *seq : seqs) {
        CHECK_ARGUMENT(seq != nullptr, "BlockManager::prepareDecode: seq pointer must not be null");
        CHECK_ARGUMENT(seq->blockSize() == block_size,
                       "BlockManager::prepareDecode: block_size mismatch");
        CHECK_ARGUMENT(seq->blockTable().size() == seq->numBlocks(),
                       "BlockManager::prepareDecode: sequence must have allocated block_table");

        batch.input_ids.push_back(seq->lastToken());
        batch.positions.push_back(static_cast<int64_t>(seq->numTokens() - 1));
        batch.context_lens.push_back(static_cast<int32_t>(seq->numTokens()));

        const size_t last_slot = seq->blockTable().back() * block_size + seq->lastBlockNumTokens() - 1;
        batch.slot_mapping.push_back(static_cast<int32_t>(last_slot));

        max_block_count = std::max(max_block_count, seq->blockTable().size());
    }

    batch.block_tables.reserve(seqs.size());
    for (const SequenceState *seq : seqs) {
        std::vector<int32_t> padded(max_block_count, -1);
        for (size_t i = 0; i < seq->blockTable().size(); ++i) {
            padded[i] = static_cast<int32_t>(seq->blockTable()[i]);
        }
        batch.block_tables.push_back(std::move(padded));
    }

    return batch;
}

uint64_t BlockManager::computeHash(const std::vector<int64_t> &token_ids, std::optional<uint64_t> prefix_hash) {
    uint64_t h = prefix_hash.value_or(1469598103934665603ull);
    for (int64_t token_id : token_ids) {
        const uint64_t x = static_cast<uint64_t>(token_id);
        h ^= std::hash<uint64_t>{}(x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        h *= 1099511628211ull;
    }
    return h;
}

void BlockManager::retain(size_t block_id) {
    CHECK_ARGUMENT(block_id < _blocks.size(), "BlockManager: block_id out of range");
    _blocks[block_id].ref_count += 1;
}

void BlockManager::release(size_t block_id) {
    CHECK_ARGUMENT(block_id < _blocks.size(), "BlockManager: block_id out of range");
    CHECK_ARGUMENT(_blocks[block_id].ref_count > 0, "BlockManager: releasing unowned block");
    _blocks[block_id].ref_count -= 1;
    if (_blocks[block_id].ref_count == 0) {
        _allocator.release(block_id);
    }
}

void BlockManager::invalidateCache(size_t block_id) {
    CHECK_ARGUMENT(block_id < _blocks.size(), "BlockManager: block_id out of range");
    auto &block = _blocks[block_id];
    if (block.hash.has_value()) {
        auto it = _hash_to_block_id.find(*block.hash);
        if (it != _hash_to_block_id.end() && it->second == block_id) {
            _hash_to_block_id.erase(it);
        }
    }
    block.hash.reset();
    block.token_ids.clear();
}

void BlockManager::updateCache(size_t block_id, uint64_t hash, const std::vector<int64_t> &token_ids) {
    CHECK_ARGUMENT(block_id < _blocks.size(), "BlockManager: block_id out of range");
    auto &block = _blocks[block_id];
    block.hash = hash;
    block.token_ids = token_ids;
    _hash_to_block_id[hash] = block_id;
}

bool BlockManager::canReuse(uint64_t hash, const std::vector<int64_t> &token_ids) const {
    auto it = _hash_to_block_id.find(hash);
    if (it == _hash_to_block_id.end()) {
        return false;
    }
    const auto &block = _blocks[it->second];
    return block.hash.has_value() && *block.hash == hash && block.token_ids == token_ids;
}

size_t BlockManager::cachedBlockId(uint64_t hash) const {
    auto it = _hash_to_block_id.find(hash);
    CHECK_ARGUMENT(it != _hash_to_block_id.end(), "BlockManager: cached block hash not found");
    return it->second;
}

void BlockManager::finalizeLastFullBlock(SequenceState &seq) {
    CHECK_ARGUMENT(seq.numTokens() % blockSize() == 0,
                   "BlockManager: finalizeLastFullBlock requires a full last block");
    CHECK_ARGUMENT(!seq.blockTable().empty(), "BlockManager: sequence must have allocated blocks");

    const size_t last_block_index = seq.numBlocks() - 1;
    const size_t block_id = seq.blockTable().at(last_block_index);
    const auto token_ids = seq.block(last_block_index);
    std::optional<uint64_t> prefix_hash;
    if (last_block_index > 0) {
        const size_t prev_block_id = seq.blockTable().at(last_block_index - 1);
        prefix_hash = _blocks[prev_block_id].hash;
    }
    const uint64_t hash = computeHash(token_ids, prefix_hash);
    invalidateCache(block_id);
    updateCache(block_id, hash, token_ids);
}

} // namespace llaisys::core::paged_kv
