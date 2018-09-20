//  Copyright (c) 2016-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "db/compaction_iteration_stats.h"
#include "db/dbformat.h"
#include "db/pinned_iterators_manager.h"
#include "db/version_edit.h"
#include "include/rocksdb/comparator.h"
#include "include/rocksdb/types.h"
#include "table/internal_iterator.h"
#include "table/scoped_arena_iterator.h"
#include "table/table_builder.h"
#include "util/kv_map.h"
#include "util/heap.h"

namespace rocksdb {

// RangeDelMaps maintain position across calls to ShouldDelete. The caller may
// wish to specify a mode to optimize positioning the iterator during the next
// call to ShouldDelete. The non-kFullScan modes are only available when
// deletion collapsing is enabled.
//
// For example, if we invoke Next() on an iterator, kForwardTraversal should be
// specified to advance one-by-one through deletions until one is found with its
// interval containing the key. This will typically be faster than doing a full
// binary search (kBinarySearch).
enum class RangeDelPositioningMode {
  kFullScan,  // used iff collapse_deletions_ == false
  kForwardTraversal,
  kBackwardTraversal,
  kBinarySearch,
};

// A RangeDelIterator iterates over range deletion tombstones.
class RangeDelIterator {
 public:
  virtual ~RangeDelIterator() = default;

  virtual bool Valid() const = 0;
  virtual void Next() = 0;
  virtual void Seek(const Slice& target) = 0;
  virtual RangeTombstone Tombstone() const = 0;
};

// A RangeDelMap keeps track of range deletion tombstones within a snapshot
// stripe.
//
// RangeDelMaps are used internally by RangeDelAggregator. They are not intended
// to be used directly.
class RangeDelMap {
 public:
  virtual ~RangeDelMap() = default;

  virtual bool ShouldDelete(const ParsedInternalKey& parsed,
                            RangeDelPositioningMode mode) = 0;
  virtual bool IsRangeOverlapped(const Slice& start, const Slice& end) = 0;
  virtual void InvalidatePosition() = 0;

  virtual size_t Size() const = 0;
  bool IsEmpty() const { return Size() == 0; }

  virtual void AddTombstone(RangeTombstone tombstone) = 0;
  virtual std::unique_ptr<RangeDelIterator> NewIterator() = 0;
};

// A RangeDelAggregator aggregates range deletion tombstones as they are
// encountered in memtables/SST files. It provides methods that check whether a
// key is covered by range tombstones or write the relevant tombstones to a new
// SST file.
class RangeDelAggregator {
 public:
  // @param snapshots These are used to organize the tombstones into snapshot
  //    stripes, which is the seqnum range between consecutive snapshots,
  //    including the higher snapshot and excluding the lower one. Currently,
  //    this is used by ShouldDelete() to prevent deletion of keys that are
  //    covered by range tombstones in other snapshot stripes. This constructor
  //    is used for writes (flush/compaction). All DB snapshots are provided
  //    such that no keys are removed that are uncovered according to any DB
  //    snapshot.
  // Note this overload does not lazily initialize Rep.
  RangeDelAggregator(const InternalKeyComparator& icmp,
                     const std::vector<SequenceNumber>& snapshots,
                     bool collapse_deletions = true);

  // @param upper_bound Similar to snapshots above, except with a single
  //    snapshot, which allows us to store the snapshot on the stack and defer
  //    initialization of heap-allocating members (in Rep) until the first range
  //    deletion is encountered. This constructor is used in case of reads (get/
  //    iterator), for which only the user snapshot (upper_bound) is provided
  //    such that the seqnum space is divided into two stripes. Only the older
  //    stripe will be used by ShouldDelete().
  RangeDelAggregator(const InternalKeyComparator& icmp,
                     SequenceNumber upper_bound,
                     bool collapse_deletions = false);

  // Returns whether the key should be deleted, which is the case when it is
  // covered by a range tombstone residing in the same snapshot stripe.
  // @param mode If collapse_deletions_ is true, this dictates how we will find
  //             the deletion whose interval contains this key. Otherwise, its
  //             value must be kFullScan indicating linear scan from beginning.
  bool ShouldDelete(
      const ParsedInternalKey& parsed,
      RangeDelPositioningMode mode = RangeDelPositioningMode::kFullScan) {
    if (rep_ == nullptr) {
      return false;
    }
    return ShouldDeleteImpl(parsed, mode);
  }
  bool ShouldDelete(
      const Slice& internal_key,
      RangeDelPositioningMode mode = RangeDelPositioningMode::kFullScan) {
    if (rep_ == nullptr) {
      return false;
    }
    return ShouldDeleteImpl(internal_key, mode);
  }
  bool ShouldDeleteImpl(const ParsedInternalKey& parsed,
                        RangeDelPositioningMode mode);
  bool ShouldDeleteImpl(const Slice& internal_key,
                        RangeDelPositioningMode mode);

  // Checks whether range deletions cover any keys between `start` and `end`,
  // inclusive.
  //
  // @param start User key representing beginning of range to check for overlap.
  // @param end User key representing end of range to check for overlap. This
  //     argument is inclusive, so the existence of a range deletion covering
  //     `end` causes this to return true.
  bool IsRangeOverlapped(const Slice& start, const Slice& end);

  // Adds tombstones to the tombstone aggregation structure maintained by this
  // object. Tombstones are truncated to smallest and largest. If smallest (or
  // largest) is null, it is not used for truncation. When adding range
  // tombstones present in an sstable, smallest and largest should be set to
  // the smallest and largest keys from the sstable file metadata. Note that
  // tombstones end keys are exclusive while largest is inclusive.
  // @return non-OK status if any of the tombstone keys are corrupted.
  Status AddTombstones(std::unique_ptr<InternalIterator> input,
                       const InternalKey* smallest = nullptr,
                       const InternalKey* largest = nullptr);

  // Resets iterators maintained across calls to ShouldDelete(). This may be
  // called when the tombstones change, or the owner may call explicitly, e.g.,
  // if it's an iterator that just seeked to an arbitrary position. The effect
  // of invalidation is that the following call to ShouldDelete() will binary
  // search for its tombstone.
  void InvalidateRangeDelMapPositions();

  bool IsEmpty();
  bool AddFile(uint64_t file_number);

  // Create a new iterator over the range deletion tombstones in all of the
  // snapshot stripes in this aggregator. Tombstones are presented in start key
  // order. Tombstones with the same start key are presented in arbitrary order.
  //
  // The iterator is invalidated after any call to AddTombstones. It is the
  // caller's responsibility to avoid using invalid iterators.
  std::unique_ptr<RangeDelIterator> NewIterator();

 private:
  // Maps snapshot seqnum -> map of tombstones that fall in that stripe, i.e.,
  // their seqnums are greater than the next smaller snapshot's seqnum.
  typedef std::map<SequenceNumber, std::unique_ptr<RangeDelMap>> StripeMap;

  struct Rep {
    StripeMap stripe_map_;
    PinnedIteratorsManager pinned_iters_mgr_;
    std::list<std::string> pinned_slices_;
    std::set<uint64_t> added_files_;
  };
  // Initializes rep_ lazily. This aggregator object is constructed for every
  // read, so expensive members should only be created when necessary, i.e.,
  // once the first range deletion is encountered.
  void InitRep(const std::vector<SequenceNumber>& snapshots);

  std::unique_ptr<RangeDelMap> NewRangeDelMap();
  RangeDelMap& GetRangeDelMap(SequenceNumber seq);

  SequenceNumber upper_bound_;
  std::unique_ptr<Rep> rep_;
  const InternalKeyComparator& icmp_;
  // collapse range deletions so they're binary searchable
  const bool collapse_deletions_;

  // TODO: move some of the definitions here to the .cc file
  // TODO: this only assumes forward iteration for now, extend to reverse
  // iteration later
  // TODO: this should be stored in a heap, and UpdatePosition is only called on
  // the top of the heap / before inserting into heap (what's passed to it?)
  //    - alternatively, it's not added to the heap until ShouldDelete is called
  class ClampedIterator {
   public:
    ClampedIterator(std::unique_ptr<InternalIterator> tombstone_iter,
                    Comparator* ucmp, InternalKey* smallest = nullptr,
                    InternalKey* largest = nullptr)
        : tombstone_iter_(std::move(tombstone_iter)),
          ucmp_(ucmp),
          smallest_(smallest),
          largest_(largest) {}

    SequenceNumber sequence() const { return cur_seqnum_; }

    Slice boundary() const { return next_range_boundary_; }

    bool Valid() const { return valid_; }

    // Must be called before adding to heap
    void UpdatePosition(const ParsedInternalKey& parsed) {
      // TODO: account for changing direction as well
      bool updated_position = false;
      if (!previously_seeked_) {
        tombstone_iter_->SeekForPrev(parsed.user_key);
        previously_seeked_ = true;
        updated_position = true;
      } else {
        // After we've seeked, we expect this to be faster
        while (tombstone_iter_->Valid() &&
               ucmp_->Compare(ExtractUserKey(tombstone_iter_->key()),
                              parsed.user_key) > 0) {
          tombstone_iter_->Next();
          updated_position = true;
        }
      }

      if (!tombstone_iter_->Valid() ||
          (largest != nullptr &&
           ucmp_->Compare(largest->user_key(), parsed.user_key) < 0)) {
        valid_ = false;
        return;
      }

      if (updated_position) {
        // TODO: check if pinned first; use a function to wrap this
        pinned_next_range_boundary_ = tombstone_iter_->value().ToString();
        next_range_boundary_ = pinned_next_range_boundary_;
        tombstone_iter_->Next();
        cur_seqnum_ = get_seqnum();
        if (tombstone_iter_->Valid() &&
            ucmp_->Compare(next_range_boundary_,
                           ExtractUserKey(tombstone_iter_->key())) > 0) {
          pinned_next_range_boundary_ =
              ExtractUserKey(tombstone_iter_->key()).ToString();
          next_range_boundary_ = pinned_next_range_boundary_;
        }
        tombstone_iter_->Prev();
      } else if (ucmp_->Compare(tombstone_iter_->value(), parsed.user_key) <=
                 0) {
        cur_seqnum_ = 0;
      }
    }

   private:
    SequenceNumber get_seqnum() const {
      return ExtractInternalKeyFooter(tombstone_iter_->key()) >> 8;
    }

    const std::unique_ptr<InternalIterator> tombstone_iter_;
    const Comparator* ucmp_;
    const InternalKey* smallest_;
    const InternalKey* largest_;

    Slice next_range_boundary_;
    // used if keys/values aren't pinned by tombstone_iter_
    std::string pinned_next_range_boundary_;
    bool previously_seeked_ = false;
    SequenceNumber cur_seqnum_ = 0;
    bool valid_ = true;
  };

  struct ClampedIteratorForwardComparator {
    ClampedIteratorComparator(const Comparator* c) : cmp(c) {}

    bool operator()(ClampedIterator* a, ClampedIterator* b) const {
      return cmp->Compare(a->boundary(), b->boundary()) > 0;
    }

    const Comparator* cmp;
  };

  typedef BinaryHeap<ClampedIterator*, ClampedIteratorForwardComparator>
      ForwardIterHeap;
  ForwardIterHeap forward_heap_;
  // TODO: maintain a heap sorted by seqnum? might use priority_queue since replace_top won't be used
  std::vector<std::unique_ptr<ClampedIterator>> iters_;
  std::vector<ClampedIterator*> new_iters_;
};

}  // namespace rocksdb
