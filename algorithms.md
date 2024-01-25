# Algorithm Design and Analysis

## Strategies

- Brute force
- Greedy
- Divide & conquer
- Randomized algorithms
- DP
- Branch and bound (Brute-force gives you all the branches, but what if you could cut some branches off early. We could use a randomly picked solution as a lower bound, in the sense that any other solution should be at least as good as a randomly picked one)


## Runtime analysis

- Big Theta notation is about **bounding** the curve of a running time of the algorithm as a function of the input size `N`
- `Big Theta of N^2` means that the running time of the algorithm as a function of the input size is a curve that can be bound **above AND below** by the curve `g(N)=c*N^2`
- Meaning, there exist such constants `c1` and `c2` that the running time curve lays strictly between `c1*N^2` and `c2*N^2`, starting from input size `N > some n`
- We need the condition `N > n` because sometimes, at very small inputs, the curve may wiggle outside the boundaries, but we are not really concerned about that (e.g. a line bounded by 2 other straight but non-parallel lines)
- The actual running time curve can be something like `3+5N+2.5N^2`, if expressed in the number of instructions (and the actual time 1 instruction takes will depend on the hardware!), but we are only interested in the bounds
- Big O notation is about the worst case analysis, i.e. the upper bound on the worst possible ("unlucky") input
- The upper bound should be a useful upper bound, meaning the lowest one. You might claim every algorithm is bounded by N^999999, but that's useless
- Similarly, Big Omega is about the lower bound, i.e. the best case analysis


## Greedy

- Maximize 1 step ahead
- The performance of a greedy algorithm depends on a selected greedy heuristic (what you maximize on)


## Divide & conquer

- Usually involves a recursion
- Split into subproblems
- You don't have to go to the smallest possible base case, you just need a small enough set so that complexity does not matter (`O(n^3)` on 10 element set is not going to matter, so you can apply brute force)
- Solve the subproblems
- Merge solutions for subproblems (the tricky part!)


## Amortized cost per operation

- **Aggregate method:** `[total cost of k ops] / k`
- This is how you get to constant amortized time for the hashtable operations, even though some ops may result in array doubling and copying
- With collections, you can delete at most the number of elements you insert, and when insert and delete have the same O, the delete operation is essentially "for free"
- Sometimes using this method is not straightforward, i.g. in case of array doubling/halving, you don't know how many inserts/deletes you may have in the future. In that case, other methods may be more convenient to use
- **Accounting method:** allow an operation "store credit in a bank account (>=0)", allow an operation to "pay for the time using credit in the bank"
- You should make sure your balance never goes negative; so the number of ops that use credit should be bounded by the number of ops that deposit
- In case of array doubling, you insert a coin every time you add an item to an array, and then you use those coins to double the array
- But you need to prove that you will have those coins when you need them, and you will not use them more than once, so that may get tricky
- **Charging method:** allow operations to charge the cost retroactively to the past (but not future). So when you calculate the cost per op, consider actual cost + total cost of all charges from the future
- In case of array doubling/halving, you know how many future inserts you need to require doubling and how many future deletes to require deletion; so you can calculate the total charge from the future
- In many cases this method is the easiest
- **Potential method:** similar to accounting method, but you define your "bank balance" as a potential function `phi` (non-negative). Think about storing potential energy that can be converted into a kinetic one
- The potential function tells you "how bad is the current state of the data structure"
- Amortized cost in this case is the `actual cost + change in phi`
- Example is binary counter, where the work can be measured in number of bits you need to change. `phi` is the number of set bits, insert at max increases `phi` by 1 (and many times actually decreases it by `t`, number of set bits at the end of the number)
- The work on increment is `1 + t`, change in `phi` is `1 - t`, so total cost is `1 + t - t + 1 = 2`, i.e. constant; in this case is very easy to calculate


## Randomized algorithms

- Depends on a random number, analyzed in terms of expectation
- **Monte Carlo:** probably correct, fixed time
- Example: verify that matrix C is a product of matrices A and B, fast and with a desired degree of probability of correctness
- **Las Vegas:** probably fast, correct result
- Example: sorting an array, will produce a correct result, but not always in exactly n*log(n) time
- If some algorithm's probability of returning a wrong answer is bound by some value <= 0.5, you can run it again and again (provided the runs are independent) to get that probability arbitrary low
- Paranoid quicksort pick the pivot element at random, then checks the size of partitions; if the partitions are not balanced enough, it repeats the partitioning
- If you define "balanced enough" as "left and right partitions should contain at least 1/4 of all elements", the probability of picking bad partition is 0.5; by repeating the process you can get the probability of getting balanced partitions arbitrary high

### Skip list

- Idea: sorted doubly-linked list L0 at the bottom, when you need to search for an element, it's `O(n)`
- But you can build a second level list L1 on top of L0, which would be similar to an express line (with fewer "stops" comparing to the bottom list)
- You "travel" as far as you can on the "express line", then "jump to the local train" and continue your "journey" there
- So your search would require `O(|L1| + |L0|/|L1|)` moves
- The expression is minimized when `|L1|` is `sqrt(|L0|)`
- So with 2 lists, search is `O(2*sqrt(n))`, with 3 lists `O(3*qbrt(n))` etc.
- Build `log(n)` levels of sorted lists to get `O(log(n))` search
- Note: once you insert on current level, you insert at all the levels below
- How to build this dynamically, when we don't know how many elements would be inserted/deleted?
- Insert: start from the bottom, insert in the current list, flip a fair coin, if heads, promote to the next level, continue until you get tails. Worst case scenario: you promote infinitely :D
- Delete: delete from the bottom level and all the levels above
- Now we need to analyze this search based on probability. We don't want a simple expectation on search, we want some bound "with high probability"
- "With high probability" means that probability grows as n grows
- For example, the number of levels, using the insert algorithm, can be shown to be bounded to `O(log(n))` with high probability. Meaning the bound is `O(c*log(n))` with probability `(1 - 1/n^alpha)`, where `alpha = c - 1`; the bigger the `n` and `c`, the narrower is the confidence interval
- When you search, you can only make as many level jumps as many levels there are. Your probability to jump level is 1/2 (corresponds to heads upon insert). The total number of levels is bounded as stated above
- So the total number of moves is a number of trials until you get a number of heads that equals number of levels (at which point you are out of levels)
- Note: as usually with probability, you could simplify this whole argument and say that expected number of horizontal moves equals number of vertical jumps (as this is determined by a fair coin flip, which is 50/50), so you could arrive to the conclusion that the expected total number of moves is `O(log(n))`, but what you lose is this notion of "with high probability"
- So instead we introduce random variables, apply Chernoff bound, and through a bunch of math be able to state that search is "O(log(n)) with high probability"

### Universal & Perfect Hashing

- Good hash function maps keys into slots evenly, meaning `Pr(h(k1) = h(k2)) = 1/m`, if `k1 != k2`, where `m` is a number of slots in a hashtable
- With simple uniform hashing analysis we assume the keys `k1` and `k2` are random, and this would be an "average case analysis"
- "Average case analysis" should be avoided, since it's unreasonable to expect the input to be random
- **Universal hashing:** choose the hash function randomly from a universe of hash functions ("universal hash family")
- The condition of universality is very similar to the criteria for the "goodness" from above, `Pr(h(k1) = h(k2)) <= 1/m`
- The difference is, now the hash function `h` is not fixed but randomly picked; on the other hand, the keys don't have to be random and can be anything
- The expectation on the number of colliding keys is now over the distribution of hash functions
- So the idea is very similar to the randomized quicksort
- There are various universal hash families to pick from
- My note: it is not completely clear how, after you picked `h` randomly, you make sure you pick the same one for the same key every time. It seems that you pick a hash function when initializing the hash table, not on every insertion. You might also re-hash if you detect the number of collisions being too big
- **Perfect hashing:** instead of linked lists for colliding items, use second level hash tables
- First level hash table hash function is picked from the universal hash family, so the first level is basically doing hashing with chaining, as usual
- Let's take `lj` to be a number of keys hashing to the same slot
- Second level hash table hash function is also picked from the universal hash family, but it maps it into the hashtable of size `lj^2`; this makes second-level collision probability very low
- To make all of this work, you need a static (known upfront) set of keys, so that you can pre-compute all this stuff (and actually start over if didn't work the first time)
- And all of this can be analyzed in terms of build time and space complexity


## DP

- Basic idea: split your problem into subproblems, solve the subproblems and then reuse solutions to the subproblems
- Example: `fib(n) = fib(n-1) + fib(n-2)` is a horrible algorithm because it is `O(2^n)`
- `fib(n-1)` and `fib(n-2)` are subproblems, but you may notice that, if were applying the algorithm, you would be solving the same subproblems again and again and again
- So instead you could memoize (remember) the solutions to subproblems, which means you would be solving subproblems in `O(1)`
- This makes the whole algorithm suddenly to be `O(n)`, since you will have exactly n non-momoized calls, each costing `O(1)`
- So in some way DP is recursion + memoization
- `Time for the problem = # of subproblems * time per subproblem`
- And count each subproblem only once
- Conversion to **bottom-up approach**: completely automatic###
- I.e. `fib[n] = fib[n-1] + fib[n-2]`, where fib is now an array (of course, take care of a base case, just like in the recursive version)
- Make sure you compute the array in the right (increasing) order, and you are good
- Since the array stores all the previously calculated values, it acts like a lookup, so no need to do memoization explicitly
- This is how you convert recursion + memoization into a table-lookup based approach
- Sometimes you don't need the whole array, like in case of fibonacci, where we only lookup into the last 2 answers
- For memoization to work, the subproblem dependencies should be acyclic


## Cache-oblivious algorithms

- In classical algorithm analysis, we assume every data access costs the same (e.g. accessing any element in an array costs the same)
- This is not true, because of the memory hierarchy and levels of caches
- Levels get bigger but at the cost of higher latency, and latency is difficult (if not impossible) to get rid of
- However it is quite possible to match the bandwidth with the size of the data layer
- So caches mitigate latency by "blocking": when fetching one word of data, get the whole block
- So you amortize the cost of data retrieval over the whole block, the cost is: `latency/block_size + 1/bandwidth`
- Typically you match the block_size with the bandwidth on the hardware side, so you can focus on latency only
- And even if you cannot control the latency, it really becomes latency amortized over the whole block, but for that you need better algorithms
- First, the algorithm need to take advantage of all the elements in the block: **spacial locality**
- Also, we want to re-use the same data elements that we already fetched for as long as possible: **temporal locality**
- The essence of the algorithm analysis from this point of view becomes estimating the number of block transfers
- The way we store data in memory and the order we access it now really matters!

### External-memory model

- It is not easy to think about multiple levels of caches at once
- External-memory model allows you to focus on just 2 levels at once
- The model: CPU with `O(1)` registers with very fast access to the cache of size `M`, divided into `N/B` blocks of size `B`, with very slow access to the disk, also divided into blocks of the same size `B`
- When you need one word in any block, the whole block is brought into the cache from the disk
- Cache access is considered for free, all we focus on is the number of blocks read/written from/to disk, and that is what we analyze
- In this model, we have explicit operations for reading/writing pages from/to disk
- Note: usually OS tries to do that transparently, but is not always doing a good job and there are software systems that allow you to have a fine control over this
- It is a bit annoying to know `M` and `B` and manually handle it, especially when `M` and `B` depend on the hardware implementation
- Also, as stated above, this model is only good for 2 levels, and it is very difficult to apply it to multiple levels of caches (and provide a good analysis)

### Cache-oblivious model

- Very similar to an external-memory model, but the algorithm is unaware of values of `M` or `B` (so it has to work well with any values of `M` and `B`)
- Since we don't know how big the blocks are, there is no explicit operations for reading/writing the block
- This means we can analyze the algorithm on all levels of caches
- My note: this probably also means this is closer to the real life, where OS actually does manage this transparently
- The whole memory space is split into logical blocks, and once you access a word in memory, the whole block is fetched (implicitly)
- When there is no more free blocks left in the cache, some blocks need to be evicted, and different strategies can be used (e.g. evict least recently used block)
- Since none of the classic algorithms we dealt before cared about `M` or `B`, all of them are cache-oblivious in this sense. But not all of them will look good _when analyzed under this model_
- So we need algorithms that are good under this model

### Example 1: scanning

- Very simple example of an algorithm: **scanning**

```
for i in range(N):
    sum += A[i]
```

- The cost (in cache-oblivious model) is `O(N/B + 1)`; +1 from the fact that you could be unlucky and go across the block boundary even when N < B
- In external-memory model, where you control the block boundaries, the cost is O(N/B)

### Example 2: parallel scan

```
for i in range(N/2):
    swap A[i] <-> A[N-i-1]
```

- We now are accessing an array from both sides, but assuming every cache level is able to fit at least 2 blocks (fair assumption), the cost is still `O(N/B + 1)`
- For more complex problems you may start analyzing whether the problem fits into cache (more than 2 blocks needed), but typically we are still more worried about the number of block transfers
