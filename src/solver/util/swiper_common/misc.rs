use crate::util::knapsack::DP;

// Returns true iff the ticket assignment in tickets is valid.
// dp_head must have exactly those indices applied that are not in indices_tail.
// indices_tail must produce distinct indices.
pub fn check_candidate<'a, I: IntoIterator<Item = &'a usize>>(
  weights: &[u64],
  tickets: &[u64],
  dp_head: &DP,
  indices_tail: I,
  adv_tickets_target: u64,
) -> bool {
  let Some(mut dp) = dp_head.make_copy(adv_tickets_target) else {
    return false;
  };

  for &index in indices_tail {
    if index < tickets.len() {
      dp = match dp.apply(weights[index], tickets[index]) {
        Some(x) => x,
        None => return false,
      };
    }
  }

  true
}
