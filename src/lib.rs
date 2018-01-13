//! A concurrent work-stealing deque.
//!
//! The data structure can be thought of as a dynamically growable and shrinkable buffer that has
//! two ends: bottom and top. A [`Deque`] can [`push`] elements into the bottom and [`pop`]
//! elements from the bottom, but it can only [`steal`][Deque::steal] elements from the top.
//!
//! A [`Deque`] doesn't implement `Sync` so it cannot be shared among multiple threads. However, it
//! can create [`Stealer`]s, and those can be easily cloned, shared, and sent to other threads.
//! [`Stealer`]s can only [`steal`][Stealer::steal] elements from the top.
//!
//! Here's a visualization of the data structure:
//!
//! ```text
//!                    top
//!                     _
//!    Deque::steal -> | | <- Stealer::steal
//!                    | |
//!                    | |
//!                    | |
//! Deque::push/pop -> |_|
//!
//!                  bottom
//! ```
//!
//! # Work-stealing schedulers
//!
//! Usually, the data structure is used in work-stealing schedulers as follows.
//!
//! There is a number of threads. Each thread owns a [`Deque`] and creates a [`Stealer`] that is
//! shared among all other threads. Alternatively, it creates multiple [`Stealer`]s - one for each
//! of the other threads.
//!
//! Then, all threads are executing in a loop. In the loop, each one attempts to [`pop`] some work
//! from its own [`Deque`]. But if it is empty, it attempts to [`steal`][Stealer::steal] work from
//! some other thread instead. When executing work (or being idle), a thread may produce more work,
//! which gets [`push`]ed into its [`Deque`].
//!
//! Of course, there are many variations of this strategy. For example, sometimes it may be
//! beneficial for a thread to always [`steal`][Deque::steal] work from the top of its deque
//! instead of calling [`pop`] and taking it from the bottom.
//!
//! # Examples
//!
//! ```
//! use crossbeam_deque::{Deque, Steal};
//! use std::thread;
//!
//! let d = Deque::new();
//! let s = d.stealer();
//!
//! d.push('a');
//! d.push('b');
//! d.push('c');
//!
//! assert_eq!(d.pop(), Some('c'));
//! drop(d);
//!
//! thread::spawn(move || {
//!     assert_eq!(s.steal(), Steal::Data('a'));
//!     assert_eq!(s.steal(), Steal::Data('b'));
//! }).join().unwrap();
//! ```
//!
//! # References
//!
//! The implementation is based on the following work:
//!
//! 1. [Chase and Lev. Dynamic circular work-stealing deque. SPAA 2005.][chase-lev]
//! 2. [Le, Pop, Cohen, and Zappa Nardelli. Correct and efficient work-stealing for weak memory
//!    models. PPoPP 2013.][weak-mem]
//! 3. [Norris and Demsky. CDSchecker: checking concurrent data structures written with C/C++
//!    atomics. OOPSLA 2013.][checker]
//!
//! [chase-lev]: https://dl.acm.org/citation.cfm?id=1073974
//! [weak-mem]: https://dl.acm.org/citation.cfm?id=2442524
//! [checker]: https://dl.acm.org/citation.cfm?id=2509514
//!
//! [`Deque`]: struct.Deque.html
//! [`Stealer`]: struct.Stealer.html
//! [`push`]: struct.Deque.html#method.push
//! [`pop`]: struct.Deque.html#method.pop
//! [Deque::steal]: struct.Deque.html#method.steal
//! [Stealer::steal]: struct.Stealer.html#method.steal

extern crate crossbeam_epoch as epoch;
extern crate crossbeam_utils as utils;

use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicIsize};
use std::sync::atomic::Ordering;

use epoch::{Atomic, Owned};
use utils::cache_padded::CachePadded;

/// Minimum buffer capacity for a deque.
const DEFAULT_MIN_CAP: usize = 1 << 4;

/// If a buffer of at least this size is retired, thread-local garbage is flushed so that it gets
/// deallocated as soon as possible.
const FLUSH_THRESHOLD_BYTES: usize = 1 << 10;

/// Possible outcomes of a steal operation.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub enum Steal<T> {
    /// The deque was empty at the time of stealing.
    Empty,

    /// Some data has been successfully stolen.
    Data(T),

    /// Lost the race for stealing data to another concurrent operation. Try again.
    Retry,
}

/// A buffer that holds elements in a deque.
struct Buffer<T> {
    /// Pointer to the allocated memory.
    ptr: *mut T,

    /// Capacity of the buffer. Always a power of two.
    cap: usize,
}

/// A macro for labeling a block.
///
/// # Examples
///
/// ```
/// fn main() {
///     let mut x = 0;
///
///     block!('l, {
///         loop {
///             x += 1;
///             break 'l;
///         }
///         panic!();
///     });
///
///     assert_eq!(x, 1);
/// }
/// ```
///
/// # Caveats
///
/// Currently, you cannot break the block with a value.
macro_rules! block {
    ($label:tt, $body:block) => ({
        $label: loop {
            $body;
            #[allow(unreachable_code)]
            { break; }
        }
    })
}

unsafe impl<T> Send for Buffer<T> {}

impl<T> Buffer<T> {
    /// Returns a new buffer with the specified capacity.
    ///
    /// # Safety
    ///
    /// `cap` should be a power of two.
    fn new(cap: usize) -> Self {
        debug_assert_eq!(cap, cap.next_power_of_two());

        let mut v = Vec::with_capacity(cap);
        let ptr = v.as_mut_ptr();
        mem::forget(v);

        Buffer {
            ptr: ptr,
            cap: cap,
        }
    }

    /// Returns a pointer to the element at the specified `index`.
    unsafe fn at(&self, index: isize) -> *mut T {
        // `self.cap` is always a power of two.
        self.ptr.offset(index & (self.cap - 1) as isize)
    }

    /// Writes `value` into the specified `index`.
    unsafe fn write(&self, index: isize, value: T) {
        ptr::write(self.at(index), value)
    }

    /// Reads a value from the specified `index`.
    unsafe fn read(&self, index: isize) -> T {
        ptr::read(self.at(index))
    }

    /// Reads values from `[from, to)`.
    unsafe fn read_range(&self, from: isize, to: isize) -> Vec<T> {
        // FIXME(jeehoonkang): it can be much more efficient by using
        // `std::slice::from_raw_parts()`.
        let mut result = Vec::new();
        let mut i = from;
        while i != to {
            result.push(ptr::read(self.at(i)));
            i = i.wrapping_add(1);
        }
        result
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            drop(Vec::from_raw_parts(self.ptr, 0, self.cap));
        }
    }
}

/// Internal data that is shared between the deque and its stealers.
struct Inner<T> {
    /// The bottom index.
    bottom: AtomicIsize,

    /// The top index.
    top: AtomicIsize,

    /// The underlying buffer.
    buffer: Atomic<Buffer<T>>,
}

impl<T> Inner<T> {
    /// Returns a new `Inner` with minimum capacity of `min_cap`.
    ///
    /// # Safety
    ///
    /// `min_cap` should be a power of two.
    fn with_capacity(cap: usize) -> Self {
        Inner {
            bottom: AtomicIsize::new(0),
            top: AtomicIsize::new(0),
            buffer: Atomic::new(Buffer::new(cap)),
        }
    }

    /// Resizes the internal buffer to the new capacity of `new_cap`.
    #[cold]
    unsafe fn resize(&self, new_cap: usize) {
        // Load the bottom, top, and buffer.
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);

        let buffer = self.buffer.load(Ordering::Relaxed, epoch::unprotected());

        // Allocate a new buffer.
        let new = Buffer::new(new_cap);

        // Copy data from the old buffer to the new one.
        let mut i = t;
        while i != b {
            ptr::copy_nonoverlapping(buffer.deref().at(i), new.at(i), 1);
            i = i.wrapping_add(1);
        }

        let guard = &epoch::pin();

        // Store the new buffer.
        self.buffer
            .store(Owned::new(new).into_shared(guard), Ordering::Release);

        // Destroy the old buffer later.
        guard.defer(move || buffer.into_owned());

        // If the buffer is very large, then flush the thread-local garbage in order to
        // deallocate it as soon as possible.
        if mem::size_of::<T>() * new_cap >= FLUSH_THRESHOLD_BYTES {
            guard.flush();
        }
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // Load the bottom, top, and buffer.
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);

        unsafe {
            let buffer = self.buffer.load(Ordering::Relaxed, epoch::unprotected());

            // Go through the buffer from top to bottom and drop all elements in the deque.
            let mut i = t;
            while i != b {
                ptr::drop_in_place(buffer.deref().at(i));
                i = i.wrapping_add(1);
            }

            // Free the memory allocated by the buffer.
            drop(buffer.into_owned());
        }
    }
}

/// A concurrent work-stealing deque.
///
/// A deque has two ends: bottom and top. Elements can be [`push`]ed into the bottom and [`pop`]ped
/// from the bottom. The top end is special in that elements can only be stolen from it using the
/// [`steal`][Deque::steal] method.
///
/// # Stealers
///
/// While [`Deque`] doesn't implement `Sync`, it can create [`Stealer`]s using the method
/// [`stealer`][stealer], and those can be easily shared among multiple threads. [`Stealer`]s can
/// only [`steal`][Stealer::steal] elements from the top end of the deque.
///
/// # Capacity
///
/// The data structure is dynamically grows as elements are inserted and removed from it. If the
/// internal buffer gets full, a new one twice the size of the original is allocated. Similarly,
/// if it is less than a quarter full, a new buffer half the size of the original is allocated.
///
/// In order to prevent frequent resizing (reallocations may be costly), it is possible to specify
/// a large minimum capacity for the deque by calling [`Deque::with_min_capacity`]. This
/// constructor will make sure that the internal buffer never shrinks below that size.
///
/// # Examples
///
/// ```
/// use crossbeam_deque::{Deque, Steal};
///
/// let d = Deque::with_min_capacity(1000);
/// let s = d.stealer();
///
/// d.push('a');
/// d.push('b');
/// d.push('c');
///
/// assert_eq!(d.pop(), Some('c'));
/// assert_eq!(d.steal(), Steal::Data('a'));
/// assert_eq!(s.steal(), Steal::Data('b'));
/// ```
///
/// [`Deque`]: struct.Deque.html
/// [`Stealer`]: struct.Stealer.html
/// [`push`]: struct.Deque.html#method.push
/// [`pop`]: struct.Deque.html#method.pop
/// [stealer]: struct.Deque.html#method.stealer
/// [`Deque::with_min_capacity`]: struct.Deque.html#method.with_min_capacity
/// [Deque::steal]: struct.Deque.html#method.steal
/// [Stealer::steal]: struct.Stealer.html#method.steal
pub struct Deque<T> {
    inner: Arc<CachePadded<Inner<T>>>,

    /// Minimum capacity of the buffer. Always a power of two.
    min_cap: usize,

    /// The maximum value of `bottom` after the last steal from the worker.
    max_bottom: Cell<isize>,

    _marker: PhantomData<*mut ()>, // !Send + !Sync
}

unsafe impl<T: Send> Send for Deque<T> {}

impl<T> Deque<T> {
    /// Returns a new deque.
    ///
    /// The internal buffer is destructed as soon as the deque and all its stealers get dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// let d = Deque::<i32>::new();
    /// ```
    pub fn new() -> Deque<T> {
        Self::with_min_capacity(DEFAULT_MIN_CAP)
    }

    /// Returns a new deque with the specified minimum capacity.
    ///
    /// If the capacity is not a power of two, it will be rounded up to the next one.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// // The minimum capacity will be rounded up to 1024.
    /// let d = Deque::<i32>::with_min_capacity(1000);
    /// ```
    pub fn with_min_capacity(min_cap: usize) -> Deque<T> {
        let power = min_cap.next_power_of_two();
        assert!(power >= min_cap, "capacity too large: {}", min_cap);
        Deque {
            inner: Arc::new(CachePadded::new(Inner::with_capacity(power))),
            min_cap: power,
            max_bottom: Cell::new(0),
            _marker: PhantomData,
        }
    }

    /// Returns `true` if the deque is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// let d = Deque::new();
    /// assert!(d.is_empty());
    /// d.push("foo");
    /// assert!(!d.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the deque.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// let d = Deque::new();
    /// d.push('a');
    /// d.push('b');
    /// d.push('c');
    /// assert_eq!(d.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        let t = self.inner.top.load(Ordering::Relaxed);
        let b = self.inner.bottom.load(Ordering::Relaxed);
        b.wrapping_sub(t).max(0) as usize
    }

    /// Pushes an element into the bottom of the deque.
    ///
    /// If the internal buffer is full, a new one twice the capacity of the current one will be
    /// allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// let d = Deque::new();
    /// d.push(1);
    /// d.push(2);
    /// ```
    pub fn push(&self, value: T) {
        unsafe {
            // Load the bottom, top, and buffer. The buffer doesn't have to be epoch-protected
            // because the current thread (the worker) is the only one that grows and shrinks it.
            let b = self.inner.bottom.load(Ordering::Relaxed);
            let t = self.inner.top.load(Ordering::Acquire);

            // Calculate the length of the deque.
            let len = b.wrapping_sub(t);

            // Calculate the capacity of the deque.
            let mut buffer = self.inner
                .buffer
                .load(Ordering::Relaxed, epoch::unprotected());
            let cap = buffer.deref().cap;

            // If the deque is full, grow the underlying buffer.
            if len >= cap as isize {
                self.inner.resize(2 * cap);
                buffer = self.inner
                    .buffer
                    .load(Ordering::Relaxed, epoch::unprotected());
            }

            // Write `value` into the right slot and increment `b`.
            buffer.deref().write(b, value);
            let b_new = b.wrapping_add(1);
            self.inner.bottom.store(b_new, Ordering::Release);

            // If `max_bottom < bottom`, then set `max_bottom = bottom`.
            if (self.max_bottom.get().wrapping_sub(b_new) as isize) < 0 {
                self.max_bottom.set(b_new);
            }
        }
    }

    /// Pushes elements into the bottom of the deque.
    ///
    /// If the internal buffer is not big enough, a new one with the capacity of the next power of
    /// two of the new size will be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// let d = Deque::new();
    /// d.push_many(vec![1, 2]);
    /// d.push_many(vec![3, 4]);
    /// ```
    pub fn push_many(&self, values: Vec<T>) {
        unsafe {
            // Load the bottom, top, and buffer. The buffer doesn't have to be epoch-protected
            // because the current thread (the worker) is the only one that grows and shrinks it.
            let b = self.inner.bottom.load(Ordering::Relaxed);
            let t = self.inner.top.load(Ordering::Acquire);

            // Calculate the length of the deque.
            let len = b.wrapping_sub(t);
            let len_values = values.len() as isize;
            let len_new = len + len_values;

            // Calculate the capacity of the deque.
            let mut buffer = self.inner
                .buffer
                .load(Ordering::Relaxed, epoch::unprotected());
            let cap = buffer.deref().cap;

            // If the deque is full, grow the underlying buffer.
            if len_new > cap as isize {
                self.inner.resize((len_new as usize).next_power_of_two() as usize);
                buffer = self.inner
                    .buffer
                    .load(Ordering::Relaxed, epoch::unprotected());
            }

            // Write `value` into the right slot and increment `b`.
            for (i, value) in values.into_iter().enumerate() {
                buffer.deref().write(b + i as isize, value);
            }
            let b_new = b.wrapping_add(len_values);
            self.inner.bottom.store(b_new, Ordering::Release);

            // If `max_bottom < bottom`, then set `max_bottom = bottom`.
            if (self.max_bottom.get().wrapping_sub(b_new) as isize) < 0 {
                self.max_bottom.set(b_new);
            }
        }
    }

    /// Pops an element from the bottom of the deque.
    ///
    /// If the internal buffer is less than a quarter full, a new buffer half the capacity of the
    /// current one will be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// let d = Deque::new();
    /// d.push(1);
    /// d.push(2);
    ///
    /// assert_eq!(d.pop(), Some(2));
    /// assert_eq!(d.pop(), Some(1));
    /// assert_eq!(d.pop(), None);
    /// ```
    pub fn pop(&self) -> Option<T> {
        // Load the bottom.
        let b = self.inner.bottom.load(Ordering::Relaxed);

        // If the deque is empty, return early without incurring the cost of a SeqCst fence.
        let t = self.inner.top.load(Ordering::Relaxed);
        if b.wrapping_sub(t) <= 0 {
            return None;
        }

        // Decrement the bottom.
        let b_new = b.wrapping_sub(1);
        self.inner.bottom.store(b_new, Ordering::Relaxed);

        atomic::fence(Ordering::SeqCst);

        // Load the buffer. The buffer doesn't have to be epoch-protected because the current
        // thread (the worker) is the only one that grows and shrinks it.
        let buffer = unsafe {
            self.inner
                .buffer
                .load(Ordering::Relaxed, epoch::unprotected())
        };
        let cap = unsafe { buffer.deref().cap };

        // Load the top.
        let mut t = self.inner.top.load(Ordering::Relaxed);

        // Calculate the length of the deque.
        let mut len = b.wrapping_sub(t);

        block!('irregular, {
            // Compute the maximum index that can be stolen.
            let max_bottom = self.max_bottom.get();
            let max_steal = t.wrapping_add(max_bottom.wrapping_add(1).wrapping_sub(t) / 2);

            // If the last element is safe to pop, go to the regular path.
            if b.wrapping_sub(max_steal) > 0 {
                break 'irregular;
            }

            // Concurrent stealers may steal the last element. Try to make a fresh consensus on
            // `bottom` with stealers by updating `top`. Concretely:
            //
            // - Atomically adds to `top` two times the length of the buffer. Note that the
            //   physical index in the buffer doesn't change.
            // - Adds to `bottom` the same amount.
            // - Sets `max_bottom` as the current value of `bottom`. It is safe because later
            //   stealers have to see it.
            // - Goes to the regular path.
            while 2 <= len {
                // FIXME(jeehoonkang): We can issue a `Release` fence before the while loop instead
                // of issuing `Release` CAS every time. We need a performance benchmark for
                // measuring its efficiency.
                match self.inner.top.compare_exchange_weak(
                    t,
                    t.wrapping_add(2 * cap as isize),
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let b_new = b_new.wrapping_add(2 * cap as isize);
                        self.inner.bottom.store(b_new, Ordering::Relaxed);
                        self.max_bottom.set(b_new);
                        break 'irregular;
                    }
                    Err(t_cur) => {
                        // Retry with a more recent value from `top`.
                        t = t_cur;
                        len = b.wrapping_sub(t);
                    }
                }
            }

            // The "irregular" path: `len <= 1`. If `len = 1`, join the race to steal from the top
            // end. If `len <= 0` or you lost the race, the deque is empty.
            //
            // FIXME(jeehoonkang): In order to also linearize the `steal()` invocations that return
            // `Empty`, the failure ordering of the CAS below should be `Acquire`. However, it
            // doesn't seem very important to do so: if your `steal()` invocation to a deque returns
            // `Empty`, you will probably retry.
            if len == 1 &&
                self.inner.top.compare_exchange(
                    t,
                    t.wrapping_add(1),
                    Ordering::Release,
                    Ordering::Relaxed).is_ok()
            {
                // Restore the bottom back to the original value at the end.
                self.inner.bottom.store(b, Ordering::Relaxed);

                // Since the worker successfully updated `top`, set `max_bottom` as the current
                // value of `bottom`.
                self.max_bottom.set(b);

                return Some(unsafe { buffer.deref().read(t) });
            }

            // Since `bottom <= top`, the deque is empty. Restore the bottom back to the original
            // value at the end, and return `None`.
            self.inner.bottom.store(b, Ordering::Relaxed);
            return None;
        });

        // The "regular" path: `steal_max < b`. It is safe to pop from the bottom end.
        unsafe {
            // Shrink the buffer if its length is less than one fourth of its capacity.
            if cap > self.min_cap && len < cap as isize / 4 {
                self.inner.resize(cap / 2);
            }

            Some(buffer.deref().read(b_new))
        }
    }

    /// Steals an element from the top of the deque.
    ///
    /// Unlike most methods in concurrent data structures, if another operation gets in the way
    /// while attempting to steal data, this method will return immediately with [`Steal::Retry`]
    /// instead of retrying.
    ///
    /// If the internal buffer is less than a quarter full, a new buffer half the capacity of the
    /// current one will be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::{Deque, Steal};
    ///
    /// let d = Deque::new();
    /// d.push(1);
    /// d.push(2);
    ///
    /// // Attempt to steal an element.
    /// //
    /// // No other threads are working with the deque, so this time we know for sure that we
    /// // won't get `Steal::Retry` as the result.
    /// assert_eq!(d.steal(), Steal::Data(1));
    ///
    /// // Attempt to steal an element, but keep retrying if we get `Retry`.
    /// loop {
    ///     match d.steal() {
    ///         Steal::Empty => panic!("should steal something"),
    ///         Steal::Data(data) => {
    ///             assert_eq!(data, 2);
    ///             break;
    ///         }
    ///         Steal::Retry => {}
    ///     }
    /// }
    /// ```
    ///
    /// [`Steal::Retry`]: enum.Steal.html#variant.Retry
    pub fn steal(&self) -> Steal<T> {
        // Load the top.
        let t = self.inner.top.load(Ordering::Relaxed);

        // Load the bottom.
        let b = self.inner.bottom.load(Ordering::Acquire);

        // Calculate the length of the deque.
        let len = b.wrapping_sub(t);

        // Is the deque empty?
        if len <= 0 {
            return Steal::Empty;
        }

        // Try to increment `top` to steal the value.
        if self.inner
            .top
            .compare_exchange_weak(t, t.wrapping_add(1), Ordering::Release, Ordering::Relaxed)
            .is_err()
        {
            return Steal::Retry;
        }

        let buffer = unsafe {
            self.inner
                .buffer
                .load(Ordering::Acquire, epoch::unprotected())
        };
        let value = unsafe { buffer.deref().read(t) };

        // Since the worker successfully updated `top`, set `max_bottom` as the current
        // value of `bottom`.
        self.max_bottom.set(b);

        // Shrink the buffer if `len - 1` is less than one fourth of `self.inner.min_cap`.
        unsafe {
            let cap = buffer.deref().cap;
            if cap > self.min_cap && len <= cap as isize / 4 {
                self.inner.resize(cap / 2);
            }
        }

        Steal::Data(value)
    }

    /// Creates a stealer that can be shared with other threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::{Deque, Steal};
    /// use std::thread;
    ///
    /// let d = Deque::new();
    /// d.push(1);
    /// d.push(2);
    ///
    /// let s = d.stealer();
    ///
    /// thread::spawn(move || {
    ///     assert_eq!(s.steal(), Steal::Data(1));
    /// }).join().unwrap();
    /// ```
    pub fn stealer(&self) -> Stealer<T> {
        Stealer {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T> fmt::Debug for Deque<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Deque {{ ... }}")
    }
}

impl<T> Default for Deque<T> {
    fn default() -> Deque<T> {
        Deque::new()
    }
}

/// A stealer that steals elements from the top of a deque.
///
/// The only operation a stealer can do that manipulates the deque is [`steal`].
///
/// Stealers can be cloned in order to create more of them. They also implement `Send` and `Sync`
/// so they can be easily shared among multiple threads.
///
/// [`steal`]: struct.Stealer.html#method.steal
pub struct Stealer<T> {
    inner: Arc<CachePadded<Inner<T>>>,
    _marker: PhantomData<*mut ()>, // !Send + !Sync
}

unsafe impl<T: Send> Send for Stealer<T> {}
unsafe impl<T: Send> Sync for Stealer<T> {}

impl<T> Stealer<T> {
    /// Returns `true` if the deque is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// let d = Deque::new();
    /// d.push("foo");
    ///
    /// let s = d.stealer();
    /// assert!(!d.is_empty());
    /// s.steal();
    /// assert!(d.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the deque.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::Deque;
    ///
    /// let d = Deque::new();
    /// let s = d.stealer();
    /// d.push('a');
    /// d.push('b');
    /// d.push('c');
    /// assert_eq!(s.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        let t = self.inner.top.load(Ordering::Relaxed);
        atomic::fence(Ordering::SeqCst);
        let b = self.inner.bottom.load(Ordering::Relaxed);
        std::cmp::max(b.wrapping_sub(t), 0) as usize
    }

    /// Steals an element from the top of the deque.
    ///
    /// Unlike most methods in concurrent data structures, if another operation gets in the way
    /// while attempting to steal data, this method will return immediately with [`Steal::Retry`]
    /// instead of retrying.
    ///
    /// This method will not attempt to resize the internal buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::{Deque, Steal};
    ///
    /// let d = Deque::new();
    /// let s = d.stealer();
    /// d.push(1);
    /// d.push(2);
    ///
    /// // Attempt to steal an element, but keep retrying if we get `Retry`.
    /// loop {
    ///     match d.steal() {
    ///         Steal::Empty => panic!("should steal something"),
    ///         Steal::Data(data) => {
    ///             assert_eq!(data, 1);
    ///             break;
    ///         }
    ///         Steal::Retry => {}
    ///     }
    /// }
    /// ```
    ///
    /// [`Steal::Retry`]: enum.Steal.html#variant.Retry
    pub fn steal(&self) -> Steal<T> {
        // Load the top.
        let t = self.inner.top.load(Ordering::Relaxed);

        // A SeqCst fence is needed here.
        // If the current thread is already pinned (reentrantly), we must manually issue the fence.
        // Otherwise, the following pinning will issue the fence anyway, so we don't have to.
        if epoch::is_pinned() {
            atomic::fence(Ordering::SeqCst);
        }

        let guard = &epoch::pin();

        // Load the bottom.
        let b = self.inner.bottom.load(Ordering::Acquire);

        // Calculate the length of the deque.
        let mut len = b.wrapping_sub(t);

        // If we observed the anomaly caused by `pop()`. Adjust the value of `len`.
        if len < -1 {
            let delta = (-len as usize + 1).next_power_of_two() as isize;
            len = len.wrapping_add(delta);
        }

        // Is the deque empty?
        if len <= 0 {
            return Steal::Empty;
        }

        // Load the buffer and read the data at the top.
        let buffer = self.inner.buffer.load(Ordering::Acquire, guard);
        let value = unsafe { buffer.deref().read(t) };

        // Try to increment `top` to steal the value.
        if self.inner
            .top
            .compare_exchange_weak(t, t.wrapping_add(1), Ordering::Release, Ordering::Relaxed)
            .is_err()
        {
            // We didn't steal this value. Forget it.
            mem::forget(value);
            return Steal::Retry;
        }

        Steal::Data(value)
    }

    /// Steals up to half the elements from the top of the deque.
    ///
    /// Unlike most methods in concurrent data structures, if another operation gets in the way
    /// while attempting to steal data, this method will return immediately with [`Steal::Retry`]
    /// instead of retrying.
    ///
    /// This method will not attempt to resize the internal buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_deque::{Deque, Steal};
    ///
    /// let d = Deque::new();
    /// let s = d.stealer();
    /// d.push(1);
    /// d.push(2);
    ///
    /// // Attempt to steal an element, but keep retrying if we get `Retry`.
    /// let stolen = loop {
    ///     match s.steal_many() {
    ///         Steal::Empty => break None,
    ///         Steal::Data(data) => break Some(data),
    ///         Steal::Retry => {}
    ///     }
    /// };
    /// assert_eq!(stolen, Some(vec!(1)));
    /// ```
    ///
    /// [`Steal::Retry`]: enum.Steal.html#variant.Retry
    pub fn steal_many(&self) -> Steal<Vec<T>> {
        // Load the top.
        let t = self.inner.top.load(Ordering::Relaxed);

        // A SeqCst fence is needed here.
        // If the current thread is already pinned (reentrantly), we must manually issue the fence.
        // Otherwise, the following pinning will issue the fence anyway, so we don't have to.
        if epoch::is_pinned() {
            atomic::fence(Ordering::SeqCst);
        }

        let guard = &epoch::pin();

        // Load the bottom.
        let b = self.inner.bottom.load(Ordering::Acquire);

        // Calculate the length of the deque.
        let mut len = b.wrapping_sub(t);

        // If we observed the anomaly caused by `pop()`. Adjust the value of `len`.
        if len < -1 {
            let delta = (-len as usize + 1).next_power_of_two() as isize;
            len = len.wrapping_add(delta);
        }

        // Is the deque empty?
        if len <= 0 {
            return Steal::Empty;
        }

        // Compute the maximum index that can be stolen.
        let max_steal = t.wrapping_add(b.wrapping_add(1).wrapping_sub(t) / 2);

        // Load the buffer and read the data at the top.
        let buffer = self.inner.buffer.load(Ordering::Acquire, guard);
        let values = unsafe { buffer.deref().read_range(t, max_steal) };

        // Try to increment `top` to steal the values.
        if self.inner
            .top
            .compare_exchange_weak(t, max_steal, Ordering::Release, Ordering::Relaxed)
            .is_err()
        {
            // We didn't steal this value. Forget it.
            mem::forget(values);
            return Steal::Retry;
        }

        Steal::Data(values)
    }
}

impl<T> Clone for Stealer<T> {
    /// Creates another stealer.
    fn clone(&self) -> Stealer<T> {
        Stealer {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T> fmt::Debug for Stealer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Stealer {{ ... }}")
    }
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::atomic::Ordering::SeqCst;
    use std::thread;

    use epoch;
    use self::rand::Rng;

    use super::{Deque, Steal};

    #[test]
    fn smoke() {
        let d = Deque::new();
        let s = d.stealer();
        assert_eq!(d.pop(), None);
        assert_eq!(s.steal(), Steal::Empty);
        assert_eq!(d.len(), 0);
        assert_eq!(s.len(), 0);

        d.push(1);
        assert_eq!(d.len(), 1);
        assert_eq!(s.len(), 1);
        assert_eq!(d.pop(), Some(1));
        assert_eq!(d.pop(), None);
        assert_eq!(s.steal(), Steal::Empty);
        assert_eq!(d.len(), 0);
        assert_eq!(s.len(), 0);

        d.push(2);
        assert_eq!(s.steal(), Steal::Data(2));
        assert_eq!(s.steal(), Steal::Empty);
        assert_eq!(d.pop(), None);

        d.push(3);
        d.push(4);
        d.push(5);
        assert_eq!(d.steal(), Steal::Data(3));
        assert_eq!(s.steal(), Steal::Data(4));
        assert_eq!(d.steal(), Steal::Data(5));
        assert_eq!(d.steal(), Steal::Empty);
    }

    #[test]
    fn steal_push() {
        const STEPS: usize = 50_000;

        let d = Deque::new();
        let s = d.stealer();
        let t = thread::spawn(move || {
            for i in 0..STEPS {
                loop {
                    if let Steal::Data(v) = s.steal() {
                        assert_eq!(i, v);
                        break;
                    }
                }
            }
        });

        for i in 0..STEPS {
            d.push(i);
        }
        t.join().unwrap();
    }

    #[test]
    fn push_many() {
        let d = Deque::new();
        let s = d.stealer();
        assert_eq!(d.pop(), None);
        assert_eq!(s.steal(), Steal::Empty);
        assert_eq!(d.len(), 0);
        assert_eq!(s.len(), 0);

        d.push_many(vec![1]);
        assert_eq!(d.len(), 1);
        assert_eq!(s.len(), 1);
        assert_eq!(d.pop(), Some(1));
        assert_eq!(d.pop(), None);
        assert_eq!(s.steal(), Steal::Empty);
        assert_eq!(d.len(), 0);
        assert_eq!(s.len(), 0);

        d.push_many(vec![2]);
        assert_eq!(s.steal(), Steal::Data(2));
        assert_eq!(s.steal(), Steal::Empty);
        assert_eq!(d.pop(), None);

        d.push_many(vec![3, 4, 5]);
        assert_eq!(d.steal(), Steal::Data(3));
        assert_eq!(s.steal(), Steal::Data(4));
        assert_eq!(d.steal(), Steal::Data(5));
        assert_eq!(d.steal(), Steal::Empty);
    }

    #[test]
    fn stampede() {
        const COUNT: usize = 50_000;

        let d = Deque::new();

        for i in 0..COUNT {
            d.push(Box::new(i + 1));
        }
        let remaining = Arc::new(AtomicUsize::new(COUNT));

        let threads = (0..8)
            .map(|_| {
                let s = d.stealer();
                let remaining = remaining.clone();

                thread::spawn(move || {
                    let mut last = 0;
                    while remaining.load(SeqCst) > 0 {
                        if let Steal::Data(x) = s.steal() {
                            assert!(last < *x);
                            last = *x;
                            remaining.fetch_sub(1, SeqCst);
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        let mut last = COUNT + 1;
        while remaining.load(SeqCst) > 0 {
            if let Some(x) = d.pop() {
                assert!(last > *x);
                last = *x;
                remaining.fetch_sub(1, SeqCst);
            }
        }

        for t in threads {
            t.join().unwrap();
        }
    }

    fn run_stress() {
        const COUNT: usize = 50_000;

        let d = Deque::new();
        let done = Arc::new(AtomicBool::new(false));
        let hits = Arc::new(AtomicUsize::new(0));

        let threads = (0..8)
            .map(|_| {
                let s = d.stealer();
                let done = done.clone();
                let hits = hits.clone();

                thread::spawn(move || {
                    while !done.load(SeqCst) {
                        if let Steal::Data(_) = s.steal() {
                            hits.fetch_add(1, SeqCst);
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        let mut rng = rand::thread_rng();
        let mut expected = 0;
        while expected < COUNT {
            if rng.gen_range(0, 3) == 0 {
                if d.pop().is_some() {
                    hits.fetch_add(1, SeqCst);
                }
            } else {
                d.push(expected);
                expected += 1;
            }
        }

        while hits.load(SeqCst) < COUNT {
            if d.pop().is_some() {
                hits.fetch_add(1, SeqCst);
            }
        }
        done.store(true, SeqCst);

        for t in threads {
            t.join().unwrap();
        }
    }

    #[test]
    fn stress() {
        run_stress();
    }

    #[test]
    fn stress_pinned() {
        let _guard = epoch::pin();
        run_stress();
    }

    #[test]
    fn no_starvation() {
        const COUNT: usize = 50_000;

        let d = Deque::new();
        let done = Arc::new(AtomicBool::new(false));

        let (threads, hits): (Vec<_>, Vec<_>) = (0..8)
            .map(|_| {
                let s = d.stealer();
                let done = done.clone();
                let hits = Arc::new(AtomicUsize::new(0));

                let t = {
                    let hits = hits.clone();
                    thread::spawn(move || {
                        while !done.load(SeqCst) {
                            if let Steal::Data(_) = s.steal() {
                                hits.fetch_add(1, SeqCst);
                            }
                        }
                    })
                };

                (t, hits)
            })
            .unzip();

        let mut rng = rand::thread_rng();
        let mut my_hits = 0;
        loop {
            for i in 0..rng.gen_range(0, COUNT) {
                if rng.gen_range(0, 3) == 0 && my_hits == 0 {
                    if d.pop().is_some() {
                        my_hits += 1;
                    }
                } else {
                    d.push(i);
                }
            }

            if my_hits > 0 && hits.iter().all(|h| h.load(SeqCst) > 0) {
                break;
            }
        }
        done.store(true, SeqCst);

        for t in threads {
            t.join().unwrap();
        }
    }

    #[test]
    fn destructors() {
        const COUNT: usize = 50_000;

        struct Elem(usize, Arc<Mutex<Vec<usize>>>);

        impl Drop for Elem {
            fn drop(&mut self) {
                self.1.lock().unwrap().push(self.0);
            }
        }

        let d = Deque::new();

        let dropped = Arc::new(Mutex::new(Vec::new()));
        let remaining = Arc::new(AtomicUsize::new(COUNT));
        for i in 0..COUNT {
            d.push(Elem(i, dropped.clone()));
        }

        let threads = (0..8)
            .map(|_| {
                let s = d.stealer();
                let remaining = remaining.clone();

                thread::spawn(move || {
                    for _ in 0..1000 {
                        if let Steal::Data(_) = s.steal() {
                            remaining.fetch_sub(1, SeqCst);
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        for _ in 0..1000 {
            if d.pop().is_some() {
                remaining.fetch_sub(1, SeqCst);
            }
        }

        for t in threads {
            t.join().unwrap();
        }

        let rem = remaining.load(SeqCst);
        assert!(rem > 0);
        assert_eq!(d.len(), rem);

        {
            let mut v = dropped.lock().unwrap();
            assert_eq!(v.len(), COUNT - rem);
            v.clear();
        }

        drop(d);

        {
            let mut v = dropped.lock().unwrap();
            assert_eq!(v.len(), rem);
            v.sort();
            for pair in v.windows(2) {
                assert_eq!(pair[0] + 1, pair[1]);
            }
        }
    }
}
