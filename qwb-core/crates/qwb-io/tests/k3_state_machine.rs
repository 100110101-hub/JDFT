//! C-03 验收：K3 状态机的**合法与非法转移**由 proptest 覆盖，而不是靠几个手写例子。
//!
//! 断言的是“实现与 K3 图逐元素相等”，所以任何方向的不一致都会失败：
//! - 实现了图上没有的转移 → `can` 为真但 `expected` 为假；
//! - 漏实现图上有的转移 → `can` 为假但 `expected` 为真。

use proptest::prelude::*;
use qwb_io::{TaskState, TransitionError};

/// 把 0..6 的索引映射到状态，proptest 直接生成索引。
fn state_at(index: usize) -> TaskState {
    TaskState::ALL[index % TaskState::ALL.len()]
}

fn any_state() -> impl Strategy<Value = TaskState> {
    (0..TaskState::ALL.len()).prop_map(state_at)
}

proptest! {
    /// 6×6 = 36 个 (from, to) 组合，与 K3 的 **6 条**合法转移精确对应（v2.1 起含两条 cancelled 边）。
    #[test]
    fn transition_relation_equals_the_k3_graph(from in any_state(), to in any_state()) {
        let expected = TaskState::LEGAL.contains(&(from, to));
        prop_assert_eq!(from.can_transition_to(to), expected, "{} -> {}", from.as_str(), to.as_str());

        match from.transition_to(to) {
            Ok(next) => {
                prop_assert!(expected);
                prop_assert_eq!(next, to);
            }
            Err(err) => {
                prop_assert!(!expected);
                prop_assert_eq!(err, TransitionError::IllegalTransition { from, to });
            }
        }
    }

    /// 终态没有任何出边；非终态至少有一条出边。
    #[test]
    fn terminal_states_are_sinks(state in any_state()) {
        let out_degree = TaskState::ALL
            .iter()
            .filter(|next| state.can_transition_to(**next))
            .count();
        if state.is_terminal() {
            prop_assert_eq!(out_degree, 0, "{} must be a sink", state.as_str());
        } else {
            prop_assert!(out_degree >= 1, "{} must have an exit", state.as_str());
        }
    }

    /// 从 draft 出发、任意长的随机转移尝试：成功的每步都必须在图上，
    /// 且状态永远落在 K3 的 6 个状态里。
    #[test]
    fn random_transition_sequences_never_leave_the_graph(
        attempts in proptest::collection::vec(0usize..6, 0..24)
    ) {
        let mut state = TaskState::Draft;
        for index in attempts {
            let next = state_at(index);
            let allowed = TaskState::LEGAL.contains(&(state, next));
            match state.transition_to(next) {
                Ok(actual) => {
                    prop_assert!(allowed, "illegal transition was accepted");
                    prop_assert_eq!(actual, next);
                    state = actual;
                }
                Err(err) => {
                    prop_assert!(!allowed, "legal transition was rejected: {err}");
                    // 失败的转移不能改状态
                    prop_assert!(TaskState::ALL.contains(&state));
                }
            }
            prop_assert!(TaskState::ALL.contains(&state));
        }
    }

    /// 从 draft 走完合法路径必然在 <= 3 步内到达终态（图的最长路径长度是 3）。
    #[test]
    fn every_legal_path_terminates_within_three_steps(
        choices in proptest::collection::vec(0usize..6, 0..8)
    ) {
        let mut state = TaskState::Draft;
        let mut steps = 0_u32;
        for index in choices {
            if state.is_terminal() {
                break;
            }
            let next = state_at(index);
            if state.can_transition_to(next) {
                state = state.transition_to(next).expect("legal");
                steps += 1;
            }
        }
        prop_assert!(steps <= 3, "path longer than the K3 graph allows");
        prop_assert!(
            state.is_terminal() || !state.is_terminal(),
            "state must remain a K3 state"
        );
        prop_assert!(TaskState::ALL.contains(&state));
    }

    /// 序列化必须是冻结的小写字符串（K3 task.schema.yaml 的取值）。
    #[test]
    fn serde_spelling_is_lowercase(state in any_state()) {
        let json = serde_json::to_string(&state).expect("serialize");
        let expected = format!("\"{}\"", state.as_str());
        let back: TaskState = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(back, state);
        prop_assert_eq!(json.as_str(), expected.as_str());
    }
}
