use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

const ACTIVE: u8 = 0;
const CANCELLED: u8 = 1;
const COMMITTED: u8 = 2;

#[derive(Clone, Debug)]
pub struct CancellationToken {
    inner: Arc<CancellationInner>,
}

#[derive(Debug)]
struct CancellationInner {
    state: AtomicU8,
    changed: tokio::sync::watch::Sender<bool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        let (changed, _) = tokio::sync::watch::channel(false);
        Self {
            inner: Arc::new(CancellationInner {
                state: AtomicU8::new(ACTIVE),
                changed,
            }),
        }
    }

    pub fn cancel(&self) -> bool {
        match self.inner.state.compare_exchange(
            ACTIVE,
            CANCELLED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                self.inner.changed.send_replace(true);
                true
            }
            Err(CANCELLED) => true,
            Err(COMMITTED) => false,
            Err(_) => unreachable!("unknown cancellation state"),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.state.load(Ordering::Acquire) == CANCELLED
    }

    pub fn try_commit(&self) -> bool {
        self.inner
            .state
            .compare_exchange(ACTIVE, COMMITTED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }

        let mut changed = self.inner.changed.subscribe();
        while !*changed.borrow_and_update() {
            if changed.changed().await.is_err() {
                return;
            }
        }
    }

    pub fn same_session(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for CancellationToken {
    fn eq(&self, other: &Self) -> bool {
        self.same_session(other)
    }
}

impl Eq for CancellationToken {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancellation_wakes_async_waiters() {
        let cancellation = CancellationToken::new();
        let waiter = cancellation.clone();
        let task = tokio::spawn(async move {
            waiter.cancelled().await;
        });

        cancellation.cancel();

        tokio::time::timeout(std::time::Duration::from_millis(100), task)
            .await
            .expect("cancellation waiter should wake")
            .expect("waiter task should complete");
    }

    #[test]
    fn committed_utterance_cannot_be_cancelled() {
        let cancellation = CancellationToken::new();

        assert!(cancellation.try_commit());
        assert!(!cancellation.cancel());
        assert!(!cancellation.is_cancelled());
    }
}
