// Training-only readout for privileged belief-head targets.
//
// This translation unit is linked beside the competition engine Export.cpp.
// It does not alter transitions.  The resulting private library must never be
// packaged in a submission: policy observations remain information-set safe,
// while these cards are written only to auxiliary training labels.

#include "All.h"

#ifdef _MSC_VER
#  define HIDDEN_API __declspec(dllexport)
#else
#  define HIDDEN_API __attribute__((visibility("default")))
#endif

namespace {

template <typename List>
int AppendCards(const State& state, const List& cards, int* output, int offset) {
  for (CardRef ref : cards) {
    output[offset++] = state.getCardId(ref);
  }
  return offset;
}

}  // namespace

extern "C" {

// Layout: [hand_count, deck_count, prize_count, hand..., deck..., prize...].
// A legal deck has at most 60 cards, so callers can always pass 63 ints.
HIDDEN_API int GetHiddenSnapshot(
    ApiData* data, int player, int* output, int capacity) {
  if (data == nullptr || data->apiDataType != 1 || player < 0 || player > 1) {
    return -1;
  }
  const State& state = data->state;
  const PlayerState& ps = state.players[player];
  const int required =
      3 + static_cast<int>(ps.hand.size()) + static_cast<int>(ps.deck.size()) +
      static_cast<int>(ps.prize.size());
  if (output == nullptr || capacity < required) {
    return required;
  }
  output[0] = static_cast<int>(ps.hand.size());
  output[1] = static_cast<int>(ps.deck.size());
  output[2] = static_cast<int>(ps.prize.size());
  int offset = 3;
  offset = AppendCards(state, ps.hand, output, offset);
  offset = AppendCards(state, ps.deck, output, offset);
  offset = AppendCards(state, ps.prize, output, offset);
  return offset;
}

HIDDEN_API int HiddenSnapshotAbiVersion() { return 1; }

}  // extern "C"
