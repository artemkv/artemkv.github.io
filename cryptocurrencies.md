# Cryptocurrencies

## References

- [MIT MAS.S62 Cryptocurrency Engineering and Design](https://www.youtube.com/playlist?list=PLUl4u3cNGP61KHzhg3JIJdK08JLSlcLId)

## Intro

- Things are valuable when we commonly decide that they are valuable. The fact that something is rare or backed by an institution are all factors in making that decision, but in the end what matters is our decision


## Traditional Payments

- Alice and Bob both have a bank account
- Bank has records (paper or digital), tracking the amount of money on each account: e.g. Alice has $10 and Bob have $0
- Alice wants to buy a sandwich from Bob. Alice contacts the bank, bank transfers the money, Alice tells Bob, Bob checks with his bank, sees the money and gives sandwich to Alice
- Bank has a central role in this exchange

### Pros

- Payments are digital, Alice and Bob can be anywhere in the world and still transact

### Cons

- You have to trust a bank, but banks can fail
- You have to talk to the bank, so bank has to be online at all times
- Privacy: bank sees everything, including very sensitive information
- Censure: bank has a power to refuse or accept transactions
- This is a real-life issue: many merchants have their funds frozen on PayPal because of being marked as "fraudulent"
- Delays in transactions and roadblocks for innovations: you depend on the bank to process the transactions, and you need to comply with a lot of legacy rules


## Traditional e-cash

- Alice contacts a bank and asks for a **digital representation of a coin**, in exchange to some money on Alice's account
- Banks creates a digital representation of a coin (basically a serial number, `SN`) and gives it to Alice
- It should be possible to verify the coin is indeed issued by the bank (done using digital signatures), and it should be possible to only use it once (ensured by bank records)
- Alice gives the coin to Bob
- Bob contacts the bank to verify that the coin is valid and not double-spent, and in that case gives sandwich to Alice
- This scheme brings us one step closer to peer-to-peer transactions, but all the same cons are still preserved


## Chaumian e-cach

- The scheme conceived by David Chaum in 1983
- Instead of bank choosing `SN` (digital coin), we allow Alice to generate her `SN` herself
- This `SN` should be random and long enough to make sure no one else will generate the same `SN`
- Alice then sends the `SN` to the bank to approve it
- The bank sends Alice a digital signature for her `SN`
- Except instead of sending the plain `SN`, Alice "blinds" the `SN` by adding extra randomness
- So in fact, Alice sends `b(SN)` to the bank, and bank replies with `Sig(b(SN))`
- The blind signing is designed in a way that allows Alice to remove blinding factor and obtain `Sig(SN)` and plain `SN`
- Alice then gives `Sig(SN),SN` pair to Bob, who sends this data to the bank, to verify the validity
- The bank keeps track of every `SN` used
- So the Bank is still responsible to verify the validity of `SN`, however, when Bob redeems the coin, the Bank has no means to know that the coin came from Alice
- But what if Alice gives the same coin to Bob and Charlie? How will they know which coin is valid?
- Interestingly, bank retains enough information to be able to trace back to Alice in that case (and only in that case!), creating an incentive for Alice not to do that
- So Bob cannot know if the coin was also given to Charlie, but he knows that if this happened, someone would punish Alice

### Pros

- Payments are digital, with all the previously mentioned pros
- Payments are peer-to-peer, which is good for privacy
- Double-spend detection does not have to be done at the moment you receive the coin, so you can do that offline

### Cons

- Banks still retain a lot of power and can censor withdrawals and deposits
- Banks weren't really interested in using this scheme, so it didn't gain any momentum


## Primitives for making cryptocurrencies: signatures and hashes

- Hashes and signatures are fundamental to cryptocurrencies
- Hashes are useful as pointers: think Git, which is basically a linked list of commits that reference previous commits by hashes
- Another way to use hashes is as commitments: if you have `y = hash(x)`, you can reveal `y` first and `x` later, and be able to prove that `x` relates to `y`
- In other words, you can make a promise `x` and commit to it by revealing `y`, but only reveal the `x` later, and people should be able to confirm that what you promised is indeed what you say you promised


## Elliptic curve signatures

- Bitcoin does not use RSA, but elliptic curves: `y^2 = x^3 + a*x + b`
- Bitcoin's elliptic curve is `y^2 = x^3 + 7`
- We can define a group structure on any smooth cubic curve
- A group is a set equipped with a binary operation (a "dot") that is associative, has an identity element, and is such that every element has an inverse
- Inverse: the curve is symmetrical about the `x`-axis, so given any point `P`, we can take `-P` to be the point opposite of it
- Binary operation (in this case "dot" is "`+`"): if `P` and `Q` are two points on the curve, then we can uniquely describe a third point, `P+Q`, in the following way. First, draw the line that intersects `P` and `Q`. This will generally intersect the curve at a third point, `R`. We then take `P+Q` to be `−R`, the point opposite `R` (`P+Q=-R`)
- Identity: define as identity the point `O` such that `P+O = O = O+P`
- If `P = Q` we only have one point, thus we can't define the line between them. In this case, we use the tangent line to the curve at this point as our line. So to calculate `P+P`, you use the tangent
- In most cases, the tangent will intersect a second point `R`, except if `P` happens to be an inflection point, in which case we take `R` to be `P` itself and `P+P` as simply the point opposite itself
- With this definition of a binary operation, it turns out the math of the curve works out in such way so that the group rules are satisfied
- Let's use `a`, `b` for scalar integer numbers, allowed operations: `a+b`, `a-b`, `a*b`, `a/b`
- Let's use `A`, `B` for points on the curve, allowed operations: `A+B`, `A-B`
- Allowed operations mixing scalars and points: `A*b`, `A/b` (but not `A+b` or `A-b`)
- `A*2` means: take a tangent through `A`, the point where it intersects with the curve is `-A*2`, then flip the point. Repeat this process multiple times for doubling multiple times
- If you are not multiplying by the power of two, you can combine factor-of-two results, i.e. `3*7 = 3*4 + 3*2 + 3`
- This shortcut is a key to constructing one-way function
- In a real world, you are not really drawing lines, you use some algorithm/formula to make the calculations; importantly, these operations are costly
- Now, using these allowed operations, we could construct a one-way function in a following way:
- Pick some random point `G` (a generator point)
- Pick some random 256-bit scalar `a` to be your private key
- Public key `A` is `a*G` (32-byte `x` coordinate, 32-byte `y` coordinate)
- Note: since the curve is symmetrical about the `x`-axis, you can encode `y` with only 1 bit (which would specify whether the point is above or below the `x` axis)
- `G` is public, `A` is the public key, so also public, the only thing that is a secret is `a`, the multiplier of `G`
- Having the first and final points, the only way to find out `a` is to repeatedly "dot" (in our case, "`+`") the initial point until you finally have the matching final point
- As we don't know `a`, we cannot apply our shortcut
- Turns out, given computational power of today's computers, trying to determine `a` from `A = a*G` given known values of `A` and `G` would take more time than what is the lifespan of the Earth
- Having `A = a*G` and `B = b*G`, turns out, `aB = bA = C`. `C` is a "Diffie Hellman" point, and allows securely exchanging cryptographic keys over a public channel (both Alice and Bob can compute `C`, but no one else can, so Bob and Alice can use `C` as a symmetric key)

### ECsig

- Message `m`, private key `a`, public key `A`
- Make `k`, a new random private key
- `R = k*G` (the only costly operation in signature calculation)
- `s = k - hash(m, R)a` (these are all scalars, hash is cheap)
- signature = `R, s`
- To verify: you know `G`, you receive `R`, `s`, `m`, `A`
- If you multiply `s` by `G`, you should get `G*(k - hash(m, R)a) = k*G - hash(m, R)a*G = R - hash(m, R)A`
- This last expression you can calculate, as well as `s*G`
- All you need to do is to verify that, given the numbers you get, `s*G == R - hash(m, R)A`
- So to verify, you have 2 costly operations: `s*G` and `hash*A`
- Caveat: if you use the same `R` value for different signatures, with the same public key, you reveal your private key; `k` has to be random and new every time


## Transactions

- What data do you need on a transaction?
- Who: Alice
- Amount: $5
- Payee: Bob
- Auth: `Sig_Alice(hash("Alice-$5-Bob"))` - proves that Alice initiated the transaction
- In an account-based model, you store the list of accounts and balances, a transaction is valid if there is a positive and sufficient balance on the account
- So with account-based model, you would have to check the Alice's account balance to see if Alice has funds and update Alice's and Bob's balances
- Account-based model would allow replaying transaction, as there is nothing to prevent it, so you also need some kind of unique id
- Ethereum uses account-based model
- This is NOT the way Bitcoin works
- Bitcoin has a concept of a coin
- Coins are not the same, each coin can have different nominal. E.g. you can have 1BTC coin, 3BTC, 5BTC coin etc.
- When spending, you are referring to a specific coin
- Coin can only be be spent once
- You cannot spend part of a coin, when spent, the whole coin is consumed, it splits and creates new coins
- So if you want to spend 4BTC coin, one option is to simply use 4BTC coin
- Alternatively, you can melt down the 1BTC coin and the 3BTC coin, and use it to re-mint a single 4BTC coin
- You can also melt down the 5BTC coin, and use it to re-mint one 4BTC and one 1BTC coin, spend 4BTC coin and keep 1BTC coin as a change
- But you can not somehow leave the 5BTC coin where it is and "break off" 4 BTC from it
- Bitcoin transaction describes which coins to use (**Input**), which coins to produce, and how to distribute them (**Output**)
- An input is a reference to an output from a previous transaction
- There can be multiple inputs and multiple outputs
- All outputs can be consumed separately
- **Input:** prev tx (a hash of previous transaction, to be able to find it), index (specific output in the referenced transaction), `scriptSig` (a signature script)
- Output: value (amount), `ScriptPubKey` (public key script)
- `ScriptPubKey` contains a hash of a public key of a person to whom you are sending the coin, this allows that person to redeem the coin later
- `scriptSig` contains transaction signature and the public key of the person who requests the transfer
- `ScriptPubKey` also contains operations (_My note: stack-based language, reminds me of Fort_), that specifies how the validation is done exactly. This gives Bitcoin a lot of flexibility
- To verify the transaction, the input's `scriptSig` and the referenced output's `ScriptPubKey` are evaluated. The input is authorized if `ScriptPubKey` returns true
- Normally, script would, predictably, use public key to validate the signature, but you could define script that defines unspendable output or an output that can be spent by anyone
- Transactions can include inputs from different people, provided all these people provided correct `scriptSig` for the corresponding inputs
- You are identified by your public key
- You can have many public keys or even use a new public key every time

### Cons

- More complex than account-based, you need to perform a lot of calculations to know your balance
- Every transaction is in the chain forever, and FBI is looking at it. Government can even blacklist coins


## Decentralized cryptocurrency system design

- Requirements:
- **Basic security:** no one should be able to intercept a transfer and steal funds
- **Permissionless:** anyone can join and use it
- **Authoritative transfer:** when Alice spends money, we need to make sure it is indeed Alice and not someone else, and no double-spends
- **Tamper-proof:** can't undo spends or change history in general. Once I receive the money from Alice, I should be certain Alice does not somehow undo the transaction and get her money back
- This leads to a general problem of maintaining a distributed database, which is normally solved by **distributed consensus** (Paxos, Zookeeper, Raft)
- The main idea of distributed consensus is maintaining a globally ordered log of transactions
- The global log allows every participant to see the past transactions and see whether the money was already spent
- With distributed consensus, the system can tolerate a certain amount of failing nodes: if some machines are not available, we can rely on others to give the answers
- However, these algorithms are not designed to deal with actively malicious behavior
- **Byzantine fault tolerance (BFT) distributed consensus** algorithms are designed to protect against actively malicious behavior
- "The Byzantine Generals Problem" was first formulated in 1982, so it is a very old problem
- Past solutions and algorithms relied on all identities being known upfront, they did not address the **Sybil attack**: a kind of security threat where one person tries to take over the network by creating multiple identities (accounts), nodes or computers
- However, if we want a global and truly decentralized cryptocurrency, anyone should be able to join without any sort of governance. This also means anyone could technically create as many accounts as they want
- Also meaning there can be millions of bots, and they are welcome
- This rules out any kind of solutions that are based on tax numbers or phone numbers, as those are governed by the centralized institutions
- In case of Bitcoin, we don't even know who designed the system
- The way to address the Sybil attack is to make identities costly, and the way to do that is to make you do some work
- Bitcoin is a one of the BFT implementations


## Proof of work

- Work should be time-consuming: `O(n)`
- Verification should be deterministic and require `O(1)`
- Memoryless: when the computation depends on a previous computation, the fastest player always wins. What we want is the setup in which, if I am 10% as fast as you, I should be able to still win 10% of a time
- Example: forge a lamport signature
- Basically, given a public key and 4 lamport signatures, created a message that contains some pre-defined text and a valid signature for it
- Every lamport signature reveals part of a private key, maybe a block from the first row, maybe from the second row, or maybe from both
- If for any block, both rows are revealed, you don't care what value the hash will produce for the corresponding bit
- So your goal is, basically, construct your message in a way so that when you hash it, the bits in the hash match the revealed blocks
- Having 4 signatures, and knowing that on average half of the bits of a hash function will be different, you will only care about 32 bits out of 256
- Since hash is a one way function, the only way to find such `x` is to brute-force. With 32 bits constrained, you will need to try about 2^32 messages
- This task should take about 3 min on AMD Ryzen 7 1700 CPU, using 8 cores
- The way to estimate the work, you need to look at the constraints (in this case 32 bits of hash that have to match)
- The real work in a given specific case may be less (if you are lucky), but with many attempts numbers will average out
- In the real world, who would be responsible to come up with 4 signatures, private and public keys etc.? We need something much simpler
- For example, find a partial collision with 0
- This was tried in the past to limit email spam: to be able to send an email, you have to compute a number which, appended to email header, will result in hash for the header having lower 20 bits being zeroes
- You find the number by starting with some initial value (e.g. 0) and if that does not work, you increment the number by one and try again, until you manage to get the result
- So if you send me a message, and I calculate the SHA-256 hash, and first 32 bits of that hash are zeroes, I know that you did O(2^32) work
- You can re-formulate this condition as "hash has to be less than target value `t`"
- In Bitcoin, to add a record to a ledger, you need to do work
- Transactions are already cryptographically signed, so you would not be able to forge a transaction from someone else, and re-playing some previous transaction would not help, as anyone would be able to detect the duplicate and reject it
- However, without proof of work you would be able to create and sign 2 valid **different** transactions (double-spend): one sending money to Alice and one sending money to Bob. Everyone would be able to detect double-spend, but no one would be able to tell which one is correct. The consensus would be able to eventually decide on a valid one, but you could overpower the consensus by creating millions of nodes
- In Bitcoin, 2 different nodes could find 2 blocks roughly at the same time, and it will take time for those 2 blocks to propagate through the system
- So you may end up with a **fork**: 2 blocks building off the same parent, creating 2 versions of the history. This is normal and happens all the time
- These 2 blocks cannot both be valid, as it might lead to double-spends
- However, until it's decided which side of the fork is valid, people may start building off both blocks
- But since the mining is a probabilistic process, one of the side of the fork will eventually grow bigger; it's very unrealistic for the next 2 blocks to be mined again at the same time and be appended again each on each side of the fork
- For each side of the fork, you could calculate amount of work that went into it
- The rule for consensus is: the side with most of work wins. Essentially, the valid block is the one that majority has build off
- The block that loses becomes "orphan", it gets abandoned and all the transactions need to be re-played
- Proof of work requirement makes it really difficult to overpower the consensus
- To put your side of fork in front you would have to provide proof of more work than the other side
- So even if the valid transaction gets in a block, it can get invalidated if it happens to be on the "bad side" of a fork
- Interestingly, you can proof the total amount of work done in the whole chain by looking just at 1 block (with the smallest hash) - see "hyper log log"


### Pros

- Anonymous: no pre-known key/signature, no entry requirements
- Memoryless: when you do work, you are not making progress, you chance of finding a nonce are proportional to number of attempts. This allows slower participants to still win periodically
- Scalable: it adapts to number of participants
- Non-interactive: you do all the work offline
- Tied to real world: re-writing the history has a cost, even if the majority of participants agree on it

### Cons

- Inefficient: almost every attempt to find the nonce fail. With today's scale of the network, you need a factory to find the next block, there is no way you can do it with one computer, so small players are pushed out of the game
- Uses a lot of power and natural resources. When it gets big enough, it begins to affect markets. Now it is already affecting GPU prices. In the future it may start affecting electricity prices
- Irregular: while a new block is found on average every 10 minutes, this is not deterministic, sometimes it is seconds, sometimes it is hours
- 51% attack: an attacker with 51% of total network power can re-write history. If you want to start a new network, the big existing players can come and easily overpower it
- People hate it: this leads to giant facilities doing calculations, and most of them are waste. Can we do something instead of proof of work? Can we have proof of work that is actually useful?


## Block chain

- General idea: given message `m`, nonce `r`, target `t`, `hash(m, r) = h`; the condition `h < t` must hold
- Tweaking `t` allows you to fine-tune the constraint
- Message `m` is a block in block chain, `hash(m, r) = h` is a block identifier
- Chain: `m = (prev, data, r)` where `prev = hash(m_prev, r_prev)`
- In other words, `m2 = (hash(m1, r1), data2, r2)`
- If you flip any bit in any block in the history, the whole chain breaks
- It should be statistically impossible to have 2 blocks with the same hash (a collision)
- You can, however, have 2 blocks pointing to the same parent block
- The rule is: the chain with the most work wins
- If this happens, the shorter branch gets cut off (this is called "Reorg"), and all the transactions are essentially rolled back
- Reorg normally does not happen until there is a difference of at least 6 blocks
- Also, everyone is building from what they think is the tip, so getting in front as an attack would be quite expensive
- Every block has a header, containing prev hash, merkle root, nonce and some other fields like version, difficulty etc.
- So data (payload) is a merkle root
- So basically instead of signing the whole set of transactions, you are signing the merkle root (of hashes of all transactions)
- This allows you to confirm that the transaction is in the block without sending the whole block
- Header also contains the timestamp the block was mined. The timestamp cannot be trusted and is not very accurate. Nodes usually tolerate about 2 hours of difference between their time and a timestamp of a block (forward or backwards)
- The reason for having timestamps is to adjust difficulty of mining
- The difficulty is adjusted every 2016 blocks based on the time it took to find the previous 2016 blocks. At the desired rate of one block each 10 minutes, 2016 blocks would take exactly two weeks to find
- 2 hour error per block works fine in practice
- The nonce is only 4 bytes longs, so other fields are used as an extra nonce bits. Timestamp is one of those, you can also shuffle transactions in the block to produce different merkle root and so on


## Sync process

- Bitcoin is a peer-to-peer network, with everyone being able to connect to everyone
- You start by downloading the code, and making sure you get the correct code and not some hacked version
- The software comes with hard-coded DNS seeds to find peers
- You connect to the peers (usually about 7 other nodes) and ask for block headers
- You download headers and re-construct the chain, verifying validity of all headers and proof of work
- Next you request and download the full blocks (170GB in total)
- For every transaction, you validate it and re-calculate the final coins by re-applying every transaction on its inputs to produce outputs. Essentially, you have to replay every transaction
- Once you process every transaction and delete all the redeemed inputs, you end up with about 3.2GB of UTXOs (unspent transaction output)
- Bitcoin promoted a lot of research to optimize these numbers
- Once you joined, you may and will, in turn, receive requests for blocks from new members who joined after you, and to maintain the system working, you should provide them with that data
- So you should keep at least some downloaded blocks, even if it takes up space (this is similar to torrents, you don't really want to seed, but someone has to)
- To submit a new transaction, you define inputs and outputs, sign and broadcast to peers (using Wallet)
- Peers who receive the transaction, broadcast further
- Everyone keeps accumulating those transactions (in mempool.dat)
- Eventually, someone puts them in a block and does the work
- When this happens, the transaction in that block are now "confirmed" (unless there is a fork), and the next block can be built
- When someone finds the block, they broadcast it to peers
- Every node does the validation of the blocks they receive and may accept or reject the new block
- Validation, among other things, includes check for double-spends and timestamp
- If I see a fork, I keep both blocks for a while, until I know which one is valid
- The validation rules are **consensus critical**
- Peers can blocklist each other


## Forks

- It is very tricky to release new versions of a software that updates those rules, as this can also create a fork with nodes running different versions of the software not agreeing on which side of the fork is a valid one
- A **soft fork** is a software upgrade that is backwards compatible with older versions, otherwise it is a **hard fork**
- Hard fork lead to 2 chains being active at the same time, possibly forever. This creates split in a community with part of it supporting the old version and a part of it migrating to the new one. Essentially the network splits into 2 separate ones
- This is how Bitcoin XT, Bitcoin Classic, Bitcoin Cash, Bitcoin Gold and others are created
- Interestingly, the money you had gets duplicated, as it is valid under both forks
- In case of Ethereum, the network bug was exploited and the hacker stole 50 million worth of Ether. The fork was introduced to re-write the history and return the money to their owners, but not everyone went with the change. This resulted in split into Ethereum and Ethereum Classic, with 2 valid histories. In one history, the hacker has the money and in another one, he does not
- What happens if people don't validate blocks and just build on top? More generic question is: what if miners do not agree with your rules and keep mining and building on top? Some people might reject those blocks, some people might accept those blocks. Again, it might create a fork with different parts of a community supporting different sides of the fork. One side can be re-orged, or it will lead to a permanent split
- There can be quite a few different scenarios happening, depending on how many miners adapted the new rule and how many full nodes that are not mining have adapted the new rule. This makes forks quite tricky
- And the most fun is, there is no real governance or single software, anyone can submit changes or come up with new software with new rules, it's all up to your peers to accept it or not
- Sometimes you can't predict what will create hard fork. It happened that changing the underlying db implementation for storing UTXOs (from LevelDB to BerkeleyDB) started to fail for some blocks, which introduced hard fork
- There is a lot of opinions on how the things have to be done, and different people are pulling into different directions
- Some people might try introducing "evil" forks
- The better way to introduce new rules is to use sort of feature flag to activate it once the majority has agreed on it
- When you submit the new transaction, you might not know which side of the fork it is going to be added. If it's valid for both sides, it might be added on both sides. You might want to try to re-play it and make sure it's added on both sides, since that duplicates your money
- This is the way exchanges might get screwed: imagine you deposit coin `A`, then the fork happens, you replay your transaction, withdraw your money, and suddenly you balance has coin `A` and coin `B`
- To prevent that, forks can provide replay protection, basically setting the rules in that way that no transaction can be added on both sides
- Forks are very complicated, many people are unaware of forks, and this creates legal issues


## Wallets

- Wallet is (usually) a separate piece of software responsible for sending and requesting money
- Your public key is your ID, you communicate it to other people to accept payments
- Using the same public key allows anyone to track the whole history of payments received by you
- To avoid this, you could use new public key every time
- But it would be very annoying, if you need to generate a new matching private key every time as well
- You can use the following trick, having 1 `p-P` pair, calculate single-use public key `A` as `A = P + hash(r, 1)*G` where `r` is some random number
- The corresponding private key `a` is `a = p + hash(r, 1)`
- You don't have to store `A`, `a`, or `r`. All you need is to enumerate over every possible value of `r` and generate `a-A` pairs again
- With `r` being reasonably small (up to a billion), this is quite doable with modern computers
- If `P` and `r` are compromised, you lose your privacy, but not money: people will be able to link all your transactions, but won't be able to use your money
- To request a payment, you communicate your public key hash
- To know when you are paid, you need to listen to the network and check the outputs of every transaction you see
- To pay, you create a transaction, sign and broadcast to peers
- Coin selection is NP problem. You want to minimize the size of your transaction, and since the input is the biggest part, you minimize number of inputs
- Using several different coins as inputs to produce 1 output creates a privacy issues: it links your two input accounts together
- Once you sign your transaction, you have to wait for it to get in the block to be sure you have actually spent your coin
- If you are using different devices, they will not know about each other's transactions, so you need to listen to your own UTXOs getting spent in every block

### SPV

- Simplified payment verification (SPV) allows you to verify payments without downloading all the data
- This is a way to run kind of "lightweight" Bitcoin node
- Similar to previous scenario, you connect, get all the headers and verify them
- You provide all your public keys and request other nodes if they have any transactions for those keys
- Other nodes provide you with the merkle tree proof of your transactions
- There is no real incentive for them to do it. You are kind of lazy and ask them to perform the scanning for you. But it is good for the network overall, so people do it
- Nodes, however, don't have to tell you the truth, they can omit transactions
- Providing all your public keys to random nodes allows anyone to link all your public keys and compute how much money you have
- A lot of people don't even run SPV, they just go to some website. This is very insecure
- Some people even trust some website to keep your private keys, which is completely stupid


## Fees

- The very first transaction in the block is special. The input of this transaction does not point anywhere
- This transaction is created by the miner of the block and usually is a reward to himself, it generates new coins and takes fees from all other transactions in the block
- There is a consensus rule that caps the amount of this transaction
- The mining fees get lower and lower all the time, and in around 100 years all the blocks will be mined (there is, by design, a limit of 21 million blocks)
- The transaction fee is implicit. The rule for the transaction is output amount ≤ input amount. If the output is less than input, the rest is a fee, and it goes to whoever produced the block
- So essentially the person who creates a transaction is the one who decides which fee to pay. But it is up to the nodes to accept this transaction or not
- So if fee is too low or zero, your transaction might not be accepted
- The fee is not proportional to the amount of money that you are moving
- It used to be time that fee was zero, but there was also a period in time that fees were crazily high
- In the end, it is up to miners to accept your fee or not, and they are trying to select the transactions with the highest fees
- However, selecting the transactions with the highest fees is not easy, as you must respect the dependencies and put them in order: the parent transactions must come before the children, and the parent transactions might have lower fees. So you might benefit from including low fee transaction to be able to chain high fee transaction on top of it
- Some transactions may get stuck for weeks (or months) before they are finally confirmed, if no miners want to include them in a block
- It is really tricky to determine a "good" fee to include in the transaction: you may include loo low and have to wait or may include too high without need
- If your transaction gets stuck, you might want to try to create a new transaction with high fee on top of previous one. But this behavior promotes many transactions, which will aggravate the problem in general (fewer transactions, less time to wait)
- If you have millions of transactions, and everyone free to include any transactions in the block, how do people coordinate what they mine for? They don't. The one who finds the block always wins. Since every block has to be appended to the top of the chain, every time the new block gets appended, it makes all other blocks in the process of mining invalid, as you need to use the new value for the prev hash
- It does happen that miners try to build the next block without validating the previous one, and it happened that 6-7 new blocks were added to the chain before the miners realized it was invalid. Those blocks get re-orged by the majority
- It is very important to give miners right incentives. Without block reward and only fees, miners may start fighting for the same and already mined block. If the miner is lucky and gets 2 blocks very quickly, the will re-org the last block and get the fees to themselves. In that setup, the more powerful miners will always overpower smaller ones, and eventually monopolize the network


## Transaction malleability and SegWit

- Malleability is a property of some cryptographic algorithms
- An encryption algorithm is "malleable" if it is possible to transform a ciphertext into another ciphertext which decrypts to a new plaintext
- The attacker in this case is not able to read the original encrypted message, nor decrypt the new ciphertext
- This, however, allows the attacker to tamper with the message
- In case of Bitcoin, this would apply to signatures: if the attacker could produce a new valid signature from existing signature, he could change the transaction in the way that it is still valid
- You cannot sign the signature itself recursively, so in Bitcoin, you sign the transaction with the signature field zeroed, and then you put the resulting signature into the signature field
- At the same time, you reference the previous transaction by hash, and that hash is from the whole transaction, including signature
- The signature is used to confirm that transaction didn't change, but how would you confirm that signature itself didn't change?
- Signatures are provided by 3rd party libraries, and some signature algorithms would allow certain changes in the signature while still recognizing it as valid (e.g. adding leading zeroes)
- And if you managed to change the signature, the transaction will have a new hash, so it will look like a new transaction, and you don't know which one will get in
- With the elliptic signature algorithm, there is a legal way to produce many completely different signatures for the same message, that are all valid. So you could do that when sending a transaction, and instead of one transaction, send many
- In any case, you would, of course, still refer to the same inputs, and have the same outputs, but this could create some situations that are confusing
- For example, your wallet might remember the transaction you created, but be completely unaware of another one with different signature/hash, and if that another transaction is accepted, your original transaction may show up as pending forever
- This might mislead a person into re-trying the transaction and losing money
- This is also a great annoyance for people who would like to add some new cool features on top of bitcoins (like multiple people signing transactions together), as this might introduce security issues
- The solution is the "segregated witness", which is a complicated name for a simple idea: don't use signature when calculating transaction hash
- This is very easy to implement from scratch, but it needed to be implemented in a backward compatible way
- This was implemented as a soft fork, using new tx format, using new "witness" field
- Old nodes see the empty signature (0 bytes), new nodes see the empty signature + witness field, containing an actual signature
- The output of the previous transaction must contain "anyone-can-spend" script, which is now given a new semantics: "check witness field"
- Old nodes see "anyone-can-spend", which in their world means the transaction does not require signature, so they consider the block valid. Signature can't change as there isn't one
- The transaction, however, appears to them as non-standard, so they would not relay it
- New nodes interpret "anyone-can-spend" script correctly as "check witness field", and verify the signature. Signature can change, but doesn't affect txid (hash)
- What prevents the output with "anyone-can-spend" script to actually be spent by anyone who is not the intended recipient? You could try, but if any miner accepted such a transaction, this would make the whole chain invalid from the point of view of new nodes, and once the new version gained the momentum, the majority would re-org the invalid blocks
- This change also resulted in increase of a max block size from 1M to 4M, but it's too boring to get into the details
- A group of bitcoin activists, developers, and China-based miners were unhappy with SegWit, so they created a hard fork (Bitcoin Cash)


## Alternative consensus mechanism

- People don't like "proof-of-work" mechanism (uses a lot of electricity, resources etc.)
- _My thought: although this is comparable to maintaining the police, security cameras, alarms, locks, bodyguards, weapons, the army etc.: this is only needed because there are people with malicious intent_
- _My thought: so in the way it's not the proof of work that is bad, it is people. proof of work is simply a digital version protection measures that already exist in physical reality_
- Proof of work would not be bad if we could implement it in the way that is more eco-friendly (e.g. less CO2)
- **Unique node list (UNL):** every node has an identity (private key), nodes sign the block, every node is on the list, and there is a central authority that certifies that every node is unique (belongs to a unique person). Majority rules
- So you don't know who your peers are, but they are guaranteed to be unique which guards against Sybil attack
- This is a step back from a truly decentralized system, but it is still relatively decentralized
- **Proof-of-stake:** coin holders sign the block. The idea is: people with coins have incentive to keep system trustful
- **Staking** is when you pledge your coins to be used for verifying transactions. Your coins are locked up while you stake them, but you can unstake them if you want to trade them
- When a block of transactions is ready to be processed, the cryptocurrency's proof-of-stake protocol will choose a validator node to review the block. The validator checks if the transactions in the block are accurate. If so, they add the block to the blockchain and receive crypto rewards for their contribution. However, if a validator proposes adding a block with inaccurate information, they lose some of their staked holdings as a penalty
- How to select the next node to sign? If, when you sign, you can somehow influence who signs next, you will try to make yourself the next signer. Once people start gaming the system, it quickly devolves into proof of work
- Another issue with proof of stake is that when to people chain off the same block, there is no randomness that make 2 alternative chains differ in size and potentially no incentive to build off the longest chain
- Reasoning: if there is no cost of signing the wrong block, I will just sign both; if I would pick one, I might pick the wrong one ("Nothing-at-stake" problem)
- One way to mitigate "nothing-at-stake": if you see someone signed both blocks, you are allowed to slash one off and take the reward to yourself
- So you need to carefully design incentives that encourage the "good" behavior and punish the "bad" one. But in general, it is more difficult to resolve conflicts with proof of stake
- Another issue: someone might go to the genesis block and rewrite the whole history from the very beginning ("long range attack"). It is also possible in Bitcoin, but you would need 51% of the network, and it would take from weeks to months, depending on the power of the network
- With proof of stake, it is arguably more feasible
- How do you distribute coins initially? Many new cryptocurrencies start with proof of work, and then move to proof of stake
- For example, Ethereum is moving to proof-of-stake consensus mechanism
- **Proof-of-space:** similar to proof of work, but instead of CPU, you use memory. So basically, you have to do something memory-intensive, instead of CPU-intensive
- One of way to turn CPU-intensive work into memory-intensive work is to require the kind of proof of work that can be pre-computed and stored on the disk, with the idea that it would be cheaper to invest into memory and pre-compute instead of computing on the fly
- You would need a function that cannot be parallelized, otherwise people with more CPU power would still win
- **Proof-of-idle:** you prove that you are not mining. Motivation: more people mine, less the reward. If we could all lower the rate of mining all together, we would still get the same reward, but with less electricity
- Problem with this agreement: someone who decides not to follow the agreement will rob others of their money
- _My thought: this is classical game theory setup :)_
- Solution: Alice pays Bob not to mine. But first Alice needs to know Bob can mine, so Alice asks Bob to mine for a small amount of time, to prove his capacity to mine. Alice then signs the block with 2 transactions, that are configured in the following way:
- If blocks come out fast, Alice gets her money back
- If blocks come out slow, Bob gets his bounty output
- So Bob is interested in slowing down his mining
- This looks like a cold war: you are basically threaten everyone else with your huge potential to mine, but you don't actually mine
- This is an idea from Tadge Dryja, is not used in any cryptocurrency
- The big problem is, although you don't use electricity, you would still require huge supply of microchips. So instead of OPEX it moves the problem into CAPEX space, and that is still very expensive
- But this is a fun idea
- There is huge amount of academic research in this area
