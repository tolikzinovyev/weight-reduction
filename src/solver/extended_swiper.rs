use super::util::swiper_common;
use crate::types::Ratio;
use crate::util::basic::{
  calc_adv_tickets_target, calc_max_adv_weight_from_weights,
};
use crate::util::knapsack::DP;

// Calculates the head indices and the tail indices for the current batch.
// The head indices will applied only once in the batch. The tail indices will
// be applied in each iteration of the batch.
fn calc_indices_head_tail(
  tickets_len: usize,
  deltas: &[usize],
) -> (Vec<usize>, Vec<usize>) {
  let mut indices_tail = deltas.to_vec();
  indices_tail.sort_unstable_by(|a, b| b.cmp(a));
  indices_tail.dedup();

  let mut indices_head: Vec<_> = (0..tickets_len).collect();
  for &index in &indices_tail {
    if index < indices_head.len() {
      indices_head.swap_remove(index);
    }
  }

  (indices_head, indices_tail)
}

// Calculates the DP head and the tail indices to apply in each iteration
// of the batch. Returns None iff the DP head cannot be constructed which means
// that the current batch should be skipped.
fn calc_dp_head_indices_tail(
  beta: Ratio,
  weights: &[u64],
  max_adv_weight: u64,
  deltas: &[usize],
  tickets: &swiper_common::Tickets,
) -> Option<(DP, Vec<usize>)> {
  let (indices_head, indices_tail) =
    calc_indices_head_tail(tickets.data().len(), deltas);

  let mut dp_head = DP::new(
    max_adv_weight,
    calc_adv_tickets_target(beta, tickets.total() + deltas.len() as u64),
  )
  .unwrap();
  for index in indices_head {
    dp_head = dp_head.apply(weights[index], tickets.data()[index])?;
  }

  Some((dp_head, indices_tail))
}

// Apply deltas to provided tickets and total_num_tickets. If after applying
// some number of deltas a valid ticket assignment is found, returns true with
// tickets containing the corresponding ticket assignment.
// Otherwise, returns false with tickets containing
// the new ticket assignment after applying all deltas.
fn apply_deltas(
  beta: Ratio,
  weights: &[u64],
  max_adv_weight: u64,
  deltas: &[usize],
  tickets: &mut swiper_common::Tickets,
) -> bool {
  let Some((dp_head, indices_tail)) =
    calc_dp_head_indices_tail(beta, weights, max_adv_weight, deltas, tickets)
  else {
    // We are exiting early. Apply all of the deltas before that.
    for &index in deltas {
      tickets.update(index);
    }
    return false;
  };

  for &index in deltas {
    tickets.update(index);
    if swiper_common::misc::check_candidate(
      weights,
      tickets.data(),
      &dp_head,
      &indices_tail,
      calc_adv_tickets_target(beta, tickets.total()),
    ) {
      return true;
    }
  }

  false
}

pub fn solve(alpha: Ratio, beta: Ratio, weights: &[u64]) -> Vec<u64> {
  debug_assert!(weights.is_sorted_by(|a, b| a >= b));

  let max_adv_weight = calc_max_adv_weight_from_weights(alpha, weights);

  let mut tickets = swiper_common::Tickets::new();
  let mut g = swiper_common::generate_deltas(weights, alpha);

  loop {
    let batch_size = (tickets.total() + 1).isqrt();
    let deltas: Vec<_> = (&mut g).take(batch_size as usize).collect();

    let ret =
      apply_deltas(beta, weights, max_adv_weight, &deltas, &mut tickets);
    if ret {
      return tickets.extract_data();
    }
  }
}

#[cfg(test)]
mod calc_indices_head_tail_tests {
  use super::calc_indices_head_tail;
  use test_case::test_case;

  struct TestCase<'a> {
    tickets_len: usize,
    deltas: &'a [usize],
    expected: (Vec<usize>, Vec<usize>),
  }

  #[test_case(
    TestCase {
      tickets_len: 0,
      deltas: &[0, 1],
      expected: (vec![], vec![0, 1]),
    };
    "zero_tickets"
  )]
  #[test_case(
    TestCase {
      tickets_len: 5,
      deltas: &[1, 3],
      expected: (vec![0, 2, 4], vec![1, 3]),
    };
    "multiple_tickets"
  )]
  #[test_case(
    TestCase {
      tickets_len: 5,
      deltas: &[3, 3],
      expected: (vec![0, 1, 2, 4], vec![3]),
    };
    "index_updated_multiple_times"
  )]
  #[test_case(
    TestCase {
      tickets_len: 5,
      deltas: &[0, 4],
      expected: (vec![1, 2, 3], vec![0, 4]),
    };
    "first_last_index_updated"
  )]
  fn all(mut test_case: TestCase) {
    let mut ret =
      calc_indices_head_tail(test_case.tickets_len, test_case.deltas);
    test_case.expected.0.sort_unstable();
    test_case.expected.1.sort_unstable();
    ret.0.sort_unstable();
    ret.1.sort_unstable();
    assert_eq!(test_case.expected, ret);
  }
}

#[cfg(test)]
mod calc_dp_head_indices_tail_tests {
  use crate::types::Ratio;
  use crate::util::knapsack::DP;

  fn calc_dp_head_indices_tail(
    beta: Ratio,
    weights: &[u64],
    max_adv_weight: u64,
    deltas: &[usize],
    tickets: &[u64],
  ) -> Option<(DP, Vec<usize>)> {
    let tickets =
      crate::solver::util::swiper_common::Tickets::from_vec(tickets.to_vec());

    super::calc_dp_head_indices_tail(
      beta,
      weights,
      max_adv_weight,
      deltas,
      &tickets,
    )
    .map(|(dp, mut indices_tail)| {
      indices_tail.sort_unstable();
      (dp, indices_tail)
    })
  }

  #[test]
  fn zero_tickets() {
    let beta = Ratio::new(1, 2);
    let weights = &[20, 30];
    let max_adv_weight = 50;
    let deltas = &[0, 1];
    let tickets = &[];

    let (dp, indices_tail) =
      calc_dp_head_indices_tail(beta, weights, max_adv_weight, deltas, tickets)
        .unwrap();

    assert_eq!(vec![0, 1], indices_tail);
    assert_eq!(0, dp.adversarial_tickets());
  }

  #[test]
  fn many_adversarial_tickets() {
    let beta = Ratio::new(1, 2);
    let weights = &[20, 30];
    let max_adv_weight = 50;
    let deltas = &[1, 2];
    let tickets = &[3];

    assert!(
      calc_dp_head_indices_tail(beta, weights, max_adv_weight, deltas, tickets)
        .is_none()
    );
  }

  #[test]
  fn basic() {
    let beta = Ratio::new(1, 2);
    let weights = &[20, 30, 10, 10];
    let max_adv_weight = 50;
    let deltas = &[2, 5];
    let tickets = &[2, 3, 9, 1];

    let (dp, indices_tail) =
      calc_dp_head_indices_tail(beta, weights, max_adv_weight, deltas, tickets)
        .unwrap();

    assert_eq!(vec![2, 5], indices_tail);
    assert_eq!(5, dp.adversarial_tickets());
  }
}
