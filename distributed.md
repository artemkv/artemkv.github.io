# Distributed Systems
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [Lindsey Kuper: CSE138 Distributed Systems](https://www.youtube.com/playlist?list=PLNPUF5QyWU8PydLG2cIJrCvnn5I_exhYx)
- [Martin Kleppmann: Distributed Systems lecture series](https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB)

## TL;DR
- Totally-ordered delivery is hard to achieve, but probably not needed, causal delivery is often good enough
- Use vector clocks for causal delivery guarantee
- You need both safety and liveness
- Strong consistency is costly, but probably not needed, causal consistency or strong convergence is often good enough
- Strong consistency requires consensus
- Paxos is a consensus protocol
- Consistent hashing allows achieving re-sharding in a minimal possible amount of movements

```
"Two nines" = 99% up = down 3.7 days/year
"Three nines" = 99.9% up = down 8.8 hours/year
"Four nines" = 99.99% up = down 53 minutes/year
"Five nines" = 99.999% up = down 5.3 minutes/year
```

## Topic map

- Ordering of events: **Lamport clocks**, **Vector clocks**
- Delivery guarantees: **FIFO**, **Causal**, **Totally-ordered**
- **FIFO delivery**: by the same process, for free when using TCP
- **Causal delivery**: vector clocks + broadcast
- **Fault models**: which kind of faults you are willing to deal with
- **Reliable delivery**: at least once
- **Replication**: the context for consistency discussion
- **Consistency as safety property**: Whatever → RYW → FIFO → Causal → Strong
- **Consensus**: when strong consistency is required
- **Consistency as liveness property**: eventual consistency
- **Sharding**: data partitioning (orthogonal to replication)


## Definition

- Distributed system is running on **several nodes** and is characterized by **partial failures** and **unbounded latency**
- Nodes are instances of software
- Failures: node crash, hardware failure of a machine, network partition, corrupted message, node is lying etc.
- Partial failure: some parts of the system are working fine, and some are not
- When you reach a certain scale, something is virtually always going to be broken, and you have to expect that and be able to work around it
- Many types of failures may be indistinguishable for participating nodes (client request is lost, timed out or the server crashed just after receiving the request)
- So when one node sends a request to another node and does not receive a response, it is impossible to know why
- The most fundamental approach to solving this problem is deciding on a timeout, a time that you are willing to wait for the response
- If you don't receive a response withing the timeout, you cannot assume failure. The side effect might have or haven't happened. So all you can do is accept the uncertainty of that situation
- This fundamental uncertainty is inevitable in a distributed system
- Why would you build a distributed system? Data too big to fit on a single machine, parallel computation, independence of failure


## Time and clocks

- Mark particular points in time (when something was created, expiration date/time)
- Measuring intervals of time (e.g. timeout)
- Computers have 2 types of physical clocks: time-of-day clocks and monotonic clocks
- **Time-of-day clocks** tell you what time it is, synchronized across machines over NTP, bad for measuring durations precisely (for things like leap seconds etc.), okeyish for timestamps
- **Monotonic clocks** only go forward, a counter (e.g. milliseconds since the machine was restarted), good for intervals and timestamps, can be used to time the code execution (`System.nanoTime()` is Java)
- In distributed systems we also need logical clocks
- **Logical clocks** only provide the ordering of events
- `A → B`: "`A` happened before `B`", important to determine potential causality

### Lamport diagrams

- In a **Lamport diagram**, events are dots on a process line that represents time from a point of view of a process
- Processes can run on different machines, every process has its own process line
- Some events are messages, i.e. "message sent", "message received", those are connected by arrows going from one process line to another
- Given two events `A` and `B`, we say that `A → B` if any of the following is true:
	1. `A` and `B` occur on the same process line with `B` after `A`
	2. `A` is a "message sent" event and `B` is a corresponding "message received" event
	3. `A → C` and `C → B` (transitive closure)
- Some events may not be related using one of the three rules, so they cannot be ordered
- In math, this kind of relationship is called "partial order"
- Formally, **partial order** on a set is an arrangement such that, for certain pairs of elements, one precedes the other
- Well, "happened before" relation is not exactly "partial order", because it does not have a "reflexivity" property, i.e. event A does not "happen before" itself (it is not clear what that would mean)

### State

- Any dot on a process line is also a state. Basically this means "any change of a state to be considered an event" (e.g. changing the value of a variable in memory)
- Consequently, the state of a process is all the events that have taken place on it
- So if you had the full sequence of events, you could fully reconstruct the final state

### Ordering of events

- Messages can arrive to a process out of order, and cause "causal anomaly" (Message "Fuck you Alice" from Bob arrives before "Bob smells" message from Alice)
- In an asynchronous network the latency is unbounded, so we cannot rely on the max delay for ordering of events
- But we can use our rules for determining the "happened before" relationship to solve this anomaly
- If we cannot infer "happened before" relation from any of 3 rules, the order between those events is undefined, and we say that the 3 events are "concurrent" or "independent" or "causally unrelated", denoted as `A || B`
- Now we need an algorithm for ordering events based on "happened before" relationship

### Lamport clocks

- Lamport clock assigns an integer to an event
- Lamport clocks guarantee that if `A → B`, then `LC(A) < LC(B)`, so we say that lamport clocks are **consistent with causality**
- Rules:
	1. Every process has a counter, initialized to 0
	2. On every event, a process will increment its counter by 1
	3. When sending a message, a process first increments the counter and then sends it along with the message
	4. When receiving a message, a process sets its counter to the `max of [local counter, the message counter] + 1`
- In some cases you may not consider "message received" as an event, then you don't need to +1 in rule 4
- Unfortunately, the rules do not work in another direction, meaning if `LC(A) < LC(B)`, we cannot say for sure that `A → B`
- We say that lamport clocks **do not characterize causality**
- So this is kind of a bummer, meaning you can't just order by `LC(X)`
- Instead, to determine whether `A` happened before `B`, you need to check whether if you can reach `A` from `B` following the lines
- Basically, causality is graph reachability in space-time
- _My thought: so instead of LC value, it sounds like you need to store prev event id, then build a graph and then analyze reachability_
- The only thing you can say using Lamport clocks is that, if `!(LC(A) < LC(B))`, then for sure `!(A → B)`. This can be useful when debugging
- That said, you can use lamport timestamps to create a total ordering of events, using some arbitrary mechanism to break ties (e.g. node name, process id)
- Creating a total order can be useful for many reasons, e.g. scheduling access to a shared resource
- However, this order is artificial, and cannot be used to imply causality

### Vector clocks

- Vector clocks guarantee that if `A → B`, then `VC(A) < VC(B)` and (the most importantly!) the implication goes in another direction too
- Vector clocks are not only consistent with causality, but they also characterize causality
- Vector clock is a vector of integers, each component corresponding to a process
- Vector clocks requires the number of processes being known and fixed upfront, as well as which component corresponds to which process
- Rules:
	1. Every process maintains a vector of integers, initialized to all zeroes; the length of a vector is a number of processes
	2. On every event, a process will increment its own component in the vector
	3. When sending a message, a process first increments its own component in the vector and then sends its current vector together with the message
	4. When receiving a message, a process first increments its own component in the vector, then updates its vector clock to the max of its own vector clock and received vector clock
- Max is "pointwise maximum", e.g. `max([1, 12, 4], [7, 0, 2])` is `[7, 12, 4]`
- Comparing vectors is also component-wise
- `VC(A) < VC(B)` when `VC(A)[i] <= VC(B)[i]` for all `i` and `VC(A) != VC(B)`
- Basically, all the components of two vectors have to be less or equal, with at least one strictly smaller
- If none of the two events is smaller than the other, they are considered to be concurrent (causally unrelated)
- So this time, to find out whether 2 events are in "happened before" relationship, instead of thinking about graph reachability all you can do is to compare 2 vectors


## Protocols and delivery guarantees

- Protocol is a set of rules that computers use to communicate with each other
- Example is "Hi, how are you?" - "Good, thanks"
- There exist violations to the protocol, such as replying "Good, thanks" without being asked or not replying
- Lamport diagrams can be useful to depict protocol violations
- In distributed systems, there is a difference between _receiving_ and _delivering_ messages
- Delivering means making a message available for processing
- Receiving is something that happens to you, delivering is what you do
- For example, if you receive messages and queue them before processing, the queue is a part of delivery mechanism
- One reason to delay message delivery after you receive a message is to make sure messages are delivered in order (FIFO delivery)
- **FIFO delivery:** if a process sends message `m2` after message `m1`, then any process delivering both messages delivers `m1` first
- Most distributed systems communicate over TCP, and FIFO delivery is a part of a protocol, so no need to implement anything on top of that
- Note that if you simply drop all the messages, you kind of guarantee FIFO delivery, but that is totally useless, of course
- **Causal delivery:** if `m1`'s send happens before `m2`'s send, then `m1`'s delivery happens before `m2`'s delivery
- The difference is FIFO delivery speaks about 2 messages sent by the same process, while causal delivery only speaks about the send order
- FIFO delivery violation is also a causal delivery violation
- Causal delivery violation is not necessarily FIFO delivery violation
- But causal delivery guarantee is also a FIFO delivery guarantee
- So causal delivery is more useful
- **Totally-ordered delivery:** if a process delivers `m1` then `m2`, then all processes delivering both `m1` and `m2` deliver `m1` first
- FIFO delivery violation is not necessarily totally-ordered delivery violation, e.g. if there is only 1 process delivering messages
- Totally-ordered delivery violation is not necessarily FIFO delivery violation
- Totally-ordered delivery guarantee does not guarantee FIFO delivery or causal delivery

### Implementing FIFO delivery

- If you are using TCP to communicate between services, then you basically have it for free, no need to implement anything
- Typical approach is to use sequence numbers
- Messages get tagged with a sequence number from the sender, and the sender id; sender increments its sequence number after sending
- The receiver checks the sequence number, and only delivers message if the sequence number is `[sequence number of the previously delivered message from the same sender] + 1`; otherwise queues it
- This would break if there is a possibility for a message to get lost, which would make all the subsequent messages queued forever
- So this only works together with reliable message delivery
- You might decide to queue the out-of-order messages for a while, and then deliver them anyway (after a certain grace period) even if there is a gap in a sequence. But then, if the lower sequence number message somehow arrives, you would need to discard it
- Another way to implement FIFO delivery is using acks
- Basically, you don't send the next message until the previously sent one get acknowledged
- Of course, this is very inefficient, so in practice you would have to do some optimizations like batching messages

### Implementing causal delivery

- Use vector clocks, only track "message send" events
- Rules:
	1. if a message sent by `P1` is delivered at `P2`, increment `P2`'s local clock in the `P1`'s position
	2. if a message is sent by a process, first increment its own position in its local clock, then include the local clock alone with the message
	3. A message sent by `P1` is only delivered at `P2` if, for the message's timestamp `T` (the vector clock attached to it), `T[P1] == local VC[P1] + 1` and `T[Pk] <= local VC[Pk]` for all `k` except `k` at `P1`
- The condition `T[P1] == local VC[P1] + 1` makes it a "next expected message" from `P1`
- The condition `T[Pk] <= local VC[Pk]` for all `k` except `k` at `P1` means "there are no messages missing from other processes"
- In other words, we ensure that the sender "does not have more information than a receiver about the other processes"
- With this approach, all sends have to be broadcasts (causal broadcast)
- The message delivery has to be reliable
- This algorithm is the one ensuring the correct sequence of messages in a messaging system ("I lost my wallet" → "found it!" → "glad to hear it" case that I was trying to solve with previous message(s) pointers)
- The causal delivery does not rule out total order anomalies: if 2 processes send messages roughly at the same time, then the 3rd process may deliver them in any order, depending on which message arrive faster
- If we want to enforce total order, you would have to do something else
- But normally, consider whether you really need to enforce total order, or you might actually be fine with simply causal delivery


## Chandy-Lamport snapshots

- Consistent global snapshots capture the global state of the system, i.e. all processes
- Uses of snapshots: checkpointing (useful initial state for when process restarts), deadlock detection (see if there is a deadlock anywhere in the system), general debugging etc.
- We cannot rely on time-of-day clock to take a snapshot at a particular time (because the clocks cannot be perfectly synchronized)
- If we did that, we could get a snapshot that "doesn't make sense", i.e. it contains event `B`, but not event `A` that is supposed to have happened before
- Tamir and Sequin Global Checkpointing Protocol allows to produce a set of consistent checkpoints; however, it is a blocking protocol (normal execution is suspended during each round of global checkpointing)
- Instead, we will use Chandy-Lamport snapshot algorithm
- Analogy: you cannot take a picture of the whole sky at once, but you don't want to miss any bird or that bird to appear on the photos twice
- For that, we need to introduce some terminology
- **Channel** is a connection from one process to another
- One channel is from `P1` to `P2`, denoted as `C12`, and another channel is from `P2` to `P1`, denoted as `C21`
- If a message was sent, but is still in flight, we say it is "in the channel"
- To start an algorithm, one process will become an "initiator process", let's say `P1` (can be any process)
- `P1` will first record its own state `S1`, and then, before doing anything else (!), send a "marker message" on all its outgoing channels
- After that, `P1` will start recording all the messages that it receives on all its incoming channels (there will be the list of messages associated with every channel)
- When a process `Pi` receives a marker message on channel `Cki`, it checks if this is a first marker message it had seen
- "Had seen" means either sent or received a marker message
- If this is a first marker message `Pi` has seen, it records its state `Si`, marks the channel `Cki` as "empty" (associate an empty list with that channel), sends a marker message on every outgoing channel `Cij`, and starts recording incoming messages on all its incoming channels except `Cki`
- If Pi has already seen a marker message before, it stops recording on the channel `Cki`, sets `Cki`'s final state as the sequence of all the incoming messages that arrived at `Cki` since recording began
- Any message coming from one of the empty channels is ignored for the purpose of a snapshot
- The marker messages themselves are not included in the snapshots
- Two processes may start initiating snapshots independently at the same time, using the same marker messages, this is not a problem
- So the Chandy-Lamport snapshot algorithm is an example of decentralized algorithms
- Marker messages can be mixed with app messages, the snapshot algorithm does not interfere with the normal execution of the processes
- In the end, we get states `Si` for every process `Pi`, and for every channel `Pji` (lists of messages for every channel `Cji` for every process `Pi`)
- It can be proven that the algorithm is guaranteed to terminate in finite time, however, the algorithm is built on multiple assumptions
- There are no failures and all messages arrive intact and only once
- The communication channels are unidirectional and FIFO ordered
- There is a communication path between any two processes in the system

### Consistent cuts

- A "cut" is a "time frontier" going across a Lamport diagram, dividing it into "past" and "future"
- An event is "in the cut" if it is on the "past side" of a cut
- A cut is consistent when for all the events `E` that are in the cut, if `F` happened before `E`, then `F` is also in the cut
- Chandy-Lamport snapshot algorithm "determines a consistent cut" (the resulting snapshot is said to be "causally correct")
- This means that if you make a cut across the process states `Si` (just at the point the marker message is sent), this cut is going to be consistent
- Additionally, if there is actually a message that goes across the cut from the past to the future, it is going to be captured in the channel recording


## Safety and liveness

- **Safety:** "something bad never happens"
- Safety properties can be violated in a finite execution (e.g. FIFO delivery)
- **Liveness:** "something good eventually happens"
- Liveness properties cannot be violated in a finite execution (e.g. Reliable delivery)
- Both properties are important; having only one of two is useless
- **Reliable delivery:** let `P1` be a process that sends a message `m` to process `P2`. If neither `P1` nor `P2` crashes, then `P2` eventually delivers `m`


## Fault models

- We could roughly classify all faults into several categories
- **Omission fault:** message gets lost (process fails to send or receive a message)
- **Timing fault:** message is slow (process responds too late), mostly ignored for the fault modeling
- **Crash fault:** process crashes or halts (stops sending or receiving messages)
- **Byzantine fault:** malicious or arbitrary behavior (process lies, drops or corrupts messages etc.)
- "Byzantine fault" terms comes from a paper on "Byzantine generals", a name picked for no particular reason
- These are not fixed categories, you could split further
- Different protocols can be designed to tolerate different faults
- Faults can be organized into a hierarchy, top to bottom: Byzantine faults → timing faults → omission faults → crash faults
- Faults higher in the hierarchy include faults lower in the hierarchy. You can say the faults lower in the hierarchy are special cases of the faults higher in the hierarchy
- By consequence, the protocol that tolerates faults higher in the hierarchy also tolerates all the faults lower in the hierarchy
- **Fault model** is a specification that specifies what kinds of faults a system may exhibit (and thus defines what kind of faults are to be tolerated by the system)
- So the omission model assumes the message can be lost, but crash model only assumes the process can crash
- The way I understand this is as follows: of course, if process can crash, it can also lose messages (by fault hierarchy), but if we assume crash model, then we ignore anything that is not a crash, leaving it to the user to deal with

### System model (a different take on fault models from Martin Kleppmann's lectures)

- There are typically 3 areas to consider: network behavior (e.g. message lost), node behavior (e.g. crashes), timing behavior (e.g. latency)
- For the network behavior, we assume bidirectional point-to-point communication between 2 nodes
- **Reliable link:** a message is received if and only if is sent. Messages still may be reordered!
- **Fair-loss link**: messages may get lost, duplicated or reordered
- **Arbitrary link**: a malicious adversary may interfere with messages (eavesdrop, modify, drop, spoof, replay)
- You can turn fair-loss link into a reliable link, using retries, deduplication etc.
- You can turn (to some degree) an arbitrary link into a fair-loss link, using cryptography, e.g. TLS
- Nodes can fail in several different ways
- **Crash-stop** (or fail-stop): node can crash at any moment, once crashed, it stops executing forever
- **Crash-recovery** (or fail-recovery): node can crash at any moment, losing its in-memory state; it may resume executing sometime later
- **Byzantine** (or fail-arbitrary): node deviates from the algorithm. Such node may do anything, including crashes or malicious behavior
- You may also make several synchrony assumptions
- **Synchronous**: upper bound on message latency, nodes execute algorithm at a known speed
- **Asynchronous**: messages can get delayed arbitrarily, nodes can pause execution arbitrarily, no timing guarantees at all
- **Partially synchronous**: asynchronous for some finite (but unknown) periods of time, synchronous otherwise
- Partially synchronous model allows to assume synchronous model for most of the time, but relax the assumptions when convenient
- Network usually have predictable latency, which can occasionally increase due to: message lost, congestion, bad configuration etc.
- Nodes usually execute code at predictable speed, with occasional pauses due to: OS scheduling issues, GC stop-the-world pauses, page swaps etc.
- When you design an algorithm, you need to be explicit about the kinds of faults that you are willing to tolerate, the faults that you are not prepared for will destroy your algorithm

### Two generals problem

- 2 generals are on the top of 2 hills with their armies, with the enemy's army being in the valley between 2 hills
- To beat the enemy, they both need to attack at the same time
- So they need to agree on the time
- Unfortunately, their only way to communicate is by sending a messenger across the valley
- Sometimes the messenger gets caught, sometimes not
- If the messenger is caught, the message is lost
- Imagine Alice sends message "attack at dawn"
- Before she attacks, she needs to make sure Bob gets the message
- So Bob sends `ack` message
- But Bob needs to make sure Alice gets `ack` message
- So Alice sends her `ack` message to Bob's `ack` message
- You can see where it is going: the infinite exchange of acks, never being able to know for sure when it is safe to attack
- So in the omission model, it's impossible for Bob and Alice to know when they can attack
- Workaround: make a plan in advance, before splitting (formally, "common knowledge")
- **Common knowledge** of some piece of information `P`:
	- Everyone knows `P`
	- Everyone knows that everyone knows `P`
	- Everyone knows that everyone knows... and so on, infinitely
- Another workaround, probability-based: Alice keeps sending the same message until getting an `ack`, after which she stops sending. This does not make it completely safe, but the more Bob waits, the more certain he gets about Alice getting an `ack`

### The Byzantine generals problem

- The version of the problem with 3 generals, where some (up to `f`) generals might be traitors; the goal, for all honest generals, is to agree on time to attack
- Honest generals don't know who the malicious ones are, but malicious generals may collude
- There is a theorem that proves that you can tolerate `f` malicious generals with `3f+1` total generals (i.e. <1/3 generals have to be malicious for the problem to be solvable)
- Using digital signatures helps, but the problem remains hard


## Reliable delivery

- Implementation: Alice sends and re-sends the message until Bob sends an `ack`. Once Alice receives an `ack`, she considers the message delivered
- Problem: Bob's acks can get lost, so Bob can receive the same message more than once
- That is why reliable delivery is sometimes called "at least once delivery"
- Duplicate messages are not the problem per se, it really depends on the meaning of the message
- Idempotent messages can be delivered more than once without any bad consequences
- In math, function `f` is idempotent if `f(x) = f(f(x))`
- Fire and forget is "at most once delivery"
- "Exactly once delivery" is an impossible dream, in practice, it can be achieved by message deduplication or relying only on idempotent messages

### Reliable broadcast

- Unicast message has 1 sender and 1 receiver (1-to-1)
- Broadcast messages has 1 sender, and everyone receives (1-to-all)
- Multicast message has 1 sender and many receivers
- **Reliable broadcast:** if a **correct** process delivers a broadcast message `m`, then all **correct** processes deliver `m`
- By "correct process" we mean "not exhibiting any faults specified by the assumed fault model"
- For example, within the crash model, the crashing process would not be considered "correct"
- It would be then OK for crashing process not to deliver the message, and this would still be considered reliable broadcast
- If we assume unicast messages are implemented as a primitive in the system, broadcast messages can be simply implemented as unicast messages to all processes
- But what if sending process crashed while doing broadcast but before sending out all the messages to all the receivers?
- Algorithm that takes care of this:
	1. All processes keep a set of delivered messages in their local state
	2. When `P` wants to broadcast `m`, it unicasts `m` to all processes except itself and adds `m` to its delivered set
	3. When `P1` receives the message `m` from `P2`, then:
		- a) if `m` is in delivered set, do nothing
		- b) otherwise, unicast the message to everyone except yourself and the sender


## Replication

- Reasons: improve availability (extend the mean time to failure), fault tolerance (prevent data loss), data locality (geo-distribution, i.e. having the data close to the clients that need it), dividing up the work (spread the load across machines)
- Downsides: expensive (you need more machines), need to keep the copies consistent (often true, but in many cases you actually don't need a strong consistency)
- **Strong consistency** (informal definition): a replicated storage system is strongly consistent if clients cannot tell that the data is replicated
- Strong consistency is basically an informal term, and generally used instead of **Linearizability**, the term that has a strict definition (Herlihy and Wang 1991) but is more difficult to understand
- **Primary-backup replication** and **Chain replication** are both strongly consistent replication protocols

### Primary-backup replication

- You pick a particular replica to be Primary; all the other replicas are considered Backups
- Clients only interact with the Primary replica, both for reads and writes
- When client makes a write request, the Primary broadcasts the write to all Backup replicas, waits for the acks, and only when every Backup replica has acknowledged the write (commit point), the Primary returns the response to the client
- When client makes a read request, the Primary simply returns the result to the client
- Primary-backup replication is good for fault tolerance, but does not provide data locality or dividing up work

### Chain replication

- Writes go to the first replica (Head replica)
- When client makes a write request, the Head replica sends the request to whatever replica is next, that one sends it to whatever replica is next etc. until we reach the last replica (Tail replica)
- The tail replica acknowledges the write (commit point) and sends the reply to the client
- Reads go directly to the tail replica
- The order of replicas must be known to everyone
- Chain replication is good for fault tolerance, but does not provide data locality. But we at least split up the work between reads (go to tail) and writes (go to head)
- So if you have roughly 15% writes and 85% reads, you get an optimal throughput that beats primary-backup replication
- At the same time, chain replication has a worse write latency comparing to primary-backup replication

### Total order vs determinism

- Determinism relates multiple runs of the system to each other
- To have determining, you need to have the same outcome on every run
- With only one replica, the total order is guaranteed by design
- But even with only one replica, different runs may produce different results, depending on the order this replica decides to deliver the messages; so we don't necessarily have determinism in this case

### Less than strong consistency

- Ideally, the clients should never know there is more than one replica (strong consistency)
- There are different ways replicas can disagree, and several properties (consistency guarantees) that can be violated:
- **Read your writes (RYW)**
- Example of violation: Alice writes `x=3`, write gets acknowledged by primary, Alice reads the value of `x` and the stale replica returns 2
- **FIFO consistency** (writes done by a single process are seen by all the processes in the order they are issued)
- Example of violation: Alice writes `x=3`, write gets acknowledged by primary, Alice writes `x=5`, write gets acknowledged by primary, but the writes replicate in a wrong order, so when the Bob asks for the value of `x`, the replica returns 3
- **Causal consistency** (writes that are related by "happens before" relationship must be seen in the same causal order by all processes)
- Example of violation: Alice writes `x=3`, write gets acknowledged by primary, and replicates to replica `A`, but not replica `B`, Bob asks for the value of `x`, replica `A` responds with 3, but when Bob asks again, stale replica `B` responds with 2
- For the consistency guarantees discussed here, the stronger ones encompass the weaker ones, the order is: Whatever → RYW → FIFO → Causal → Strong
- Some people define as much as 50 levels of consistency guarantees, those do not nest nicely; but you don't necessarily need to know all of those
- It is very hard to maintain strong consistency, but you probably do not need it
- Causal consistency is often considered pretty good


## Consensus

- There are times when you really do need a strong consistency
- This happens when you have a bunch of processes, and you are trying to solve one of the following problems:
- **Totally-ordered broadcast (also known as atomic broadcast)**: all the processes need to deliver the same messages in the same order
- **Group membership:** all the processes need to know what other processes exist and keep those lists up-to-date
- **Leader election:** one of processes needs to play a particular role, and everyone else need to know about it
- **Distributed mutual exclusion:** processes need to take turns getting access to a shared resource
- **Distributed transaction**: processes need to participate in a transaction and agree whether the transaction should be committed or aborted
- And if you want strong consistency, you need consensus, because all of these problems boil down to a consensus problem
- The essence of consensus protocol for a single bit problem: many processes come each with their value of a bit (0 or 1) that can differ and leave all agreeing on the same value (whenever that is 0 or 1)
- Consensus algorithms are very costly, so you do not want to use them all the time, only for the things that matter
- **Paxos** is one of the best known protocols for consensus

### Properties of consensus

- Consensus algorithms **try** to satisfy the following 3 properties:
- Termination: each correct process eventually decides on a value
- Agreement: all correct processes decide on the same value
- Validity (aka integrity or non-triviality): the agreed-upon value must be one of the proposed values
- In asynchronous network model, it is actually impossible to satisfy all 3 properties, and this can be proven mathematically
- So usually consensus algorithms (e.g. Paxos) compromise on termination


## Paxos

- Paxos was invented, guess by whom... yes, by Laslie Lamport (1998)
- The recommended paper on Paxos is "Paxos made simple"
- Paxos has many flavors
- In Paxos, processes (nodes) can play 3 roles: proposer, acceptor and learner
- **Proposers** propose values
- **Acceptors** contribute to choosing from among the proposed values
- **Learners** learn the agreed upon value
- A single process can take multiple roles (or even all of them), it is just easier to reason about Paxos thinking 1 process can only take 1 role
- Paxos nodes must persist data
- Paxos nodes must know how many nodes is considered majority

### Paxos algorithm

#### Phase 1

- Proposer sends a `Prepare(n)` message (where `n` is a **proposal number**) to (at least) a majority of acceptors
- Proposal number is not an actual value that is being proposed, not to be confused, that number will come in phase 2
- Proposal number `n` should be a **globally unique** value, and higher than any proposal number that this proposer has used before
- Example: `Prepare(5)`
- To make numbers globally unique, you may decide that `P1` uses only even numbers, and the process `P2` only odd ones
- To make sure the number should be higher than one number used before by this process, every process needs to remember the highest number used by the process itself
- Acceptor, on receiving a `Prepare(n)` message, checks "did I previously promise to ignore requests with this proposal number?"
- If so, then ignore the message!
- If not, now reply to the proposal with a `Promise(n)` message: a promise to ignore any requests with a proposal number lower than `n`
- The milestone 1 is reached when proposer receives the `Promise(n)` message from the majority of acceptors
- Once this happens, it is impossible now to get the majority to promise anything less than `n`

#### Phase 2

- Proposer sends an `Accept(n, val)` message to (at least) a majority of acceptors (do not have to be the same acceptors as in phase 1, just need to be a majority)
- `n` is proposal number, matching the previous messages
- `val` is an actual proposed value
- Acceptor, on receiving `Accept(n, val)` message, checks "did I previously promise to ignore requests with this proposal number?"
- If so, then ignore the message!
- If not, now reply with `Accepted(n, val)` message, and send this message to all learners
- The milestone 2 (**consensus**) is reached when the majority of acceptors have sent `Accepted (n, val)` message
- When proposer receives the majority of `Accepted (n, val)` messages, it knows the `consensus` has been reached
- The milestone 3 is reached when each participant knows the consensus has been reached, happens separately on the proposer and learners

#### Paxos, corner cases

- There are situations that make the Paxos run more complicated
- For example, there can be more than 1 proposer
- If this happens, one of the proposers will get to the milestone 1
- At the same time, another proposer will sit and wait for the majority of acceptors to respond, but this will never happen, since the majority of them have already promised to ignore that `n`
- With Paxos, it is always safe to simply restart the process by sending a `Prepare(n)` message with a higher `n`
- So the proposer that didn't get the majority of replies within a certain timeout will restart the process
- Now, to take care of that situation, we need to make a slight adjustment to the acceptor's behavior
- Same as before, an acceptor, on receiving a `Prepare(n)` message, checks "did I previously promise to ignore requests with this proposal number?"
- If so, then ignore the message! This part does not change
- If not (and this is new!), check "have I previously accepted anything?"
- If so, reply with `Promise(n, (n_prev, val_prev))`
- If not, then just simply reply with `Promise(n)`
- With this new adjustment, the proposer will start receiving all of these messages
- So now, to take advantage of this new types of messages, we need to adjust the proposer's behavior too
- Proposer, instead of simply sending an `Accept(n, val)` message, sends `Accept(n, val*)` message
- `Val` is `val_prev` that went with the highest `n_prev`, in case it got any `Promise(n, (n_prev, val_prev))` messages; otherwise simply `val`

#### Dueling proposers

- When there is more than 1 proposer, then despite the adjustments we made, you can actually get to the infinite loop, when they keep proposing again and again, never getting to the consensus
- This is one of the scenarios where Paxos cannot guarantee termination

### Multi-Paxos

- If you use a regular Paxos, you can decide on only 1 value
- Multi-Paxos is useful when you want to decide on the sequence of values
- Example is totally-ordered broadcast, when you need to agree on the order of delivery of a sequence of messages
- With normal Paxos, solving this problem would require a lot of overhead
- How do avoid the overhead?
- Turns out, once you got to the phase 2, you can (safely) keep sending `Accept(n, val)` messages without need to repeat the phase 1, even with the same n
- If some other proposer kicks in, however, this will ruin the free lunch, but until that happens, the first proposer can go on
- Our hope is, this will rarely happen
- This is basically the multi-paxos

### Fault tolerance in Paxos

- Paxos is defined in a very defensive way, to be able to overcome crashes
- Since Paxos relies on majority, that is the minimal number of acceptors that need to be alive, the minority of acceptors can crash
- All but one proposer can crash
- Losing a message is not a big deal either, in the worst case, you simply restart the process by sending a new Prepare message with higher number
- Overall, Paxos falls in a category of algorithms that trade liveness for safety

### Other consensus algorithms

- **Raft**. Originates in Stanford, 2014, so quite recent. Raft is designed specifically to be easy to understand, there is nothing really innovative there otherwise, and this is why it was difficult for the authors to publish it. But it is so simple and so well explained (presumably) that you "could implement it by just reading the paper"
- **Viewstamped replication**. This protocol is from 1988, comes from MIT, inspired Raft
- **Zab (ZooKeeper atomic broadcast)**. Comes from Yahoo, late 2000s, part of ZooKeeper. ZooKeeper is open source, so you can use it
- The good paper that compares different protocols: https://www.cs.cornell.edu/fbs/publications/viveLaDifference.pdf

### Raft

- Raft is a distributed as a library


## Active vs passive replication

- Active replication is also called state machine replication
- Active replication: execute an operation of every replica ("deposit $50"), each replica calculates the new state
- Passive replication: execute operation on one replica and send the state update to other replicas ("New account balance is $70")
- Passive replication is preferred if the operation is expensive to execute
- Active replication is preferred when the size of the state is huge
- To implement active replication you can use strongly consistent replication protocols or consensus


## Eventual consistency

- **Eventual consistency:** replicas eventually agree if clients stop submitting updates
- Eventual consistency is liveness property, unlike RYW, FIFO, causal consistency, which are safety properties
- So eventual consistency is a separate thing, does not fit the same hierarchy
- This makes eventual consistency naming quite confusing
- Strong convergence is a safety property
- **Strong convergence:** replicas that have delivered the same set of updates have equivalent state
- Strong eventual consistency = Eventual consistency + Strong convergence
- Updates that are concurrent can arrive to replicas in different order, so replicas can begin to disagree on the state even despite causal delivery
- You can, of course, store all concurrent updates as a set, and let clients resolve conflicts
- **Network partition:** situation when parts of network cannot communicate (or it takes too long for them to communicate)
- Any messages that cross the partition are considered lost
- **Availability**: every request receives a response
- Situation: client sends write to primary, primary cannot reach the backup, what do you do?
- As network partitions are fact of life, you are essentially choosing between consistency and availability (CAP theorem)
- Systems like DynamoDB make design choice to prioritize availability over consistency
- Once network partition heals, the replicas catch up, eventually
- The tradeoff is not black and white, the balance between consistency and availability is tunable and based on quorum


## Testing distributed systems

- If you want to test how the system behaves in face of errors, you need to intentionally inject faults into your system (Chaos engineering)
- To find bugs more quickly, you need to be strategic about it, see "lineage-driven fault injection" paper


## DynamoDB

- In a certain way, DynamoDB paper was revolutionary
- Before the paper on Dynamo, people cared a lot about the strong consistency and byzantine fault tolerance, and suddenly DynamoDB didn't seem to care about it that much
- In case of DynamoDB, you have eventual consistency, and because of that, possibly diverging histories of updates, with application-specific conflict resolution (by default "last write wins")
- Amazon didn't invent the term "eventual consistency" but definitely popularized it
- If writes commute, then the order they are applied does not matter, so the strong convergence is achieved "for free"
- DynamoDB uses vector clocks for versioning updates, which helps to avoid some conflicts
- Anti-entropy and gossip are tools for resolving conflicts
- **Anti-entropy** deals with resolving conflicts in the application state (in the actual payload, from application point of view)
- **Gossip** deals with resolving conflicts in view state (view on group membership, i.e. "who is up?")
- Gossip: every once in a while (~1 min), a node will pick a node to ask who is alive ("what is your view?")
- Gossip requires sending a list live nodes over the wire, which is a small amount of data
- App state, on the other hand, can be gigabytes of data. So keeping it in sync can be tricky, since you don't want to send that much data back and forth
- Amazon uses merkle trees to quickly find the parts of data that differ, which allows comparing large amounts of data very efficiently
- Merkle tree is a tree of hashes, you have data at the leaf level, then hashes and then hashes of hashes etc.
- Initially, you can just send the root hash over the wire. If those are the same, then replicas agree
- If not, you send the next level hashes, and so on, until you get to the point of disagreement

### Quorum consistency

- In DynamoDB, the client can talk to any replica (there is no "primary")
- The client, in a way, plays a role of a primary replica
- In quorum systems you can configure how many replicas a client should talk to
- Examples: DynamoDB, Cassandra
- `N` is a total number of replicas
- `W` ("Write quorum") is a number of replicas that have to acknowledge a write for it to be considered completed
- `R` ("Read quorum") is a number of replicas that have to acknowledge a read for the client to consider operation completed
- `N=3`, `W=3`, `R=1` → does not give you strong consistency, bad fault tolerance, slow writes
- The suggestion from DynamoDB is to use `R + W > N`, in this case at least some replicas are going to return the most recent value (it's up to you to decide which one)
- The key is: the read quorum have to intersect with the write quorum

### Tail latency

- If you draw the distribution of latency per request, in most of the systems most of the requests will have a very small latency, and then a long tail
- The average latency is not really useful metrics in this case, since 2 systems can have pretty much the same average latency, and the difference will be in the length of a tail
- And the long tail is really not good
- What you are really interested is "how slow is the slowest response"
- More precisely, "what is the latency at the 99.9% of the distribution", and that number will tell you a lot


## Sharding

- Sometimes data just does not fit on one machine
- If everyone stores all data, consistency is more expensive to maintain
- If you are in a "primary-backup" setup, the primary may get overloaded with requests
- Sharding solves both problems
- Sharding and replication are orthogonal, and in the real world systems you will have both
- When deciding the sharding strategy, the aim is to achieve equal distribution of keys across nodes
- Of course, you should do it in a predictable way, so that when you need to fetch the data, you know to which partition to go
- Using keys directly could lead to hotspots
- The answer is, instead of using keys directly, use hash of a key, and then do `mod N` to know at which node to store the value
- The trouble begins if you want to add more nodes, since changing `N` would require re-sharding and that is really painful
- If you really need to re-shard, you want to minimize the amount of keys moved (avoiding keys being moved across old nodes)
- The minimal amount of data to be moved is `K/N`, where `K` is the number of keys (and `N` is the number of nodes)
- With `K/N` movements, the data only moves from the old nodes to the new one

### Consistent hashing

- **Consistent hashing** allows achieving re-sharding in a `K/N` movements
- With consistent hashing, all the nodes are arranged in the ring with each node belonging to a certain point in the ring
- When you need to access a key, you take a hash and then look for a node that is responsible for that range of values
- The node is responsible for all the keys coming before it
- Each key gets replicated to `M` next (available) nodes
- When you add a new node, you insert it somewhere in the ring; the only data that needs to move to the new node is the data on one side of the split, `K/N` on average
- When one of the node crashes, the successor just takes care of that node's range; and being (previously) the next replica for those keys, it will already have them
- With consistent hashing there is a risk of cascading crashes. Think one node having too much load, it goes down, which can overload the next node
- For the consistent hashing to work, the nodes have to be distributed evenly across the ring. The more nodes you have, the more likely you will have an even distribution
- In order to artificially increase the number of nodes, every physical node can be split into many virtual nodes each belonging to its point in the ring
- Virtual nodes can also help when nodes differ in sizes and storage capacity, this way you can assign more virtual nodes to more powerful nodes with more storage and vice versa
- In case of a node crash, you lose multiple virtual nodes, however, Amazon actually says it's a feature, since this allows to spread the network traffic moving the data upon re-sharding


## Offline systems and MapReduce

- **Online systems** wait for client requests and try to handle them quickly
- In online systems latency and availability is prioritized
- **Offline systems** or batch processing systems process lots of data, the priority is high throughput
- Streaming systems are kind of a hybrid between online and offline
- MapReduce is a classic example of an offline system
- The need arises when you have raw data, and you need to build different representations of it (derived data), e.g. inverted index
- MapReduce is a tool to compute derived data
- In lots of cases, the computation itself is conceptually very simple, but a lot of effort needs to be put into handling "distributed stuff"
- MapReduce is a framework that handles that "distributed stuff"
- See the details in the separate lecture notes, in short, it is map → shuffle → reduce


## The math behind replica conflict resolution

- Given a partially ordered set `S`, and upper bound of a pair `[a,b]` in `S` is an element `u` in `S` such that `a <= u` and `b <= u`
- The most interesting upper bounds are the smallest ones, aka "least upper bounds"
- An upper bound of `[a,b]` in `S` is the _least upper bound_ if `u <= v` for each upper bound `v` of `a` and `b`
- A partially ordered set `S` in which every 2 elements have a least upper bound is called a **join-semilattice**
- 2 conflicting state updates that do not have least upper bound cannot be combined, so you would need consensus to avoid conflicts
- However, if all state updates that replicas can take are the part of join-semilattice, then you have a natural way of resolving conflicts, without need to run consensus
- Example: "add book to the basket", "add blender to the basket" have a least upper bound of "add book and blender to the basket"
- **Conflict-free replicated data types** are the data types that are designed specifically for replication, forming a join-semilattice
- The work by merging updates by finding the least upper bound
