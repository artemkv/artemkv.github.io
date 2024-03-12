# Probability, random processes

## References

- [Introduction to Probability, Statistics, and Random Processes by Hossein Pishro-Nik](https://www.probabilitycourse.com/)
- [MIT 6.041 Probabilistic Systems Analysis and Applied Probability](https://www.youtube.com/playlist?list=PLUl4u3cNGP61MdtwGTqZA0MreSaDybji8)
- [Harvard Statistics 110: Probability](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)

## Bernoulli process

- Memoryless, discrete time
- A sequence of coin flips
- More formally, a sequence of independent Bernoulli trials
- At each trial, `P(success) = P(Xi = 1) = p`, `P(failure) = P(Xi = 0) = 1-p`
- We can generalize this as `P(Xi = x) = p^x*(1-p)^(1-x)` (Bernoulli distribution)
- We assume that `p` is constant
- We assume that trials are independent
- This might seem very basic, but you could model surprisingly many processes as Bernoulli, even if they don't really look as Bernoulli
- For example, you could model stock market as Bernoulli, looking, at the end of every day, whether the price went up or down
- Apparently, it is quite hard to beat this primitive model even with more sophisticated models
- Similarly, you could model a server receiving requests at random times: you split timeline in very small intervals, and at each interval the request either comes or not
- In this example, `p` is obviously not constant, as there are times of the day that you receive more traffic, but we ignore that
- We do that by saying "we will consider a fixed period of time during which the p is more or less constant"
- _My thought: at some point you might question why all this mathematical vigor, if we already ignored so many factors, why not just using some "common sense" heuristics and then simply test the model to see if it's good enough?_
- One way to think about this process is as of sequence of random variables `X1, X2, ...`
- At any moment `t` (i.e. at any trial `t`), `E[Xt] = p`, `Var(Xt) = p(1-p)`
- Since all the trials are independent, joint distribution of any subset of `X` is just a product of their distributions
- Another way to think about this process as of one single very long experiment, with the sample space composed of all possible outcomes (all possible sequences of zeroes and ones)
- In this case we might be interested in such events as "obtaining all ones, all the way to infinity"
- This particular event can be proven to have probability of zero
- In fact, obtaining any specific infinite sequence has probability of zero
- Here we are bridging into the world of continuous random variables, if you think of an infinite binary sequence as a binary representation of a real number. The experiment, from that point of view, is essentially the same as picking a real number at random

### Application

- Back to a scenario with a server receiving requests
- Questions you might want to ask:
- 1) given the amount of time, how many requests would arrive?
- 2) given the number of requests, what time would it take for them to arrive?
- Answering the question 1:
- Let `S` to be a number of successes in `n` time slots (number of requests in a given amount of time)
- This is a binomial distribution, which PMF is well-known
- `P(S=k) = (number of combinations of k elements from an n-element set) * p^k * (1-p)^(n-k)`
- "The number of..." is something that we already can count, see "Counting"
- `E[S] = n*p`
- `Var(S) = n*p(1-p)`
- _My thought: this is basically the formalized version of the "common sense" traffic estimation, doesn't sound like a big "aha" moment_
- Answering the question 2:
- Let `T1` to be the number of trials until the first success (interarrival interval)
- This is a geometric distribution, also well-known
- `P(T1 = t) = (1-p)^(t-1)*p`
- `E[T1] = 1/p`
- `Var(T1) = (1-p)/p^2`
- We continue and consider `T2`, the number of trials until the second success after the first success, `T3` until the third and so on
- The total number of trials until the third success `Y3 = T1 + T2 + T3`
- `T1`, `T2` and `T3` are all independent, and all distributed geometrically
- To calculate `P(Yk = t)` we could use a nice shortcut
- `P(Yk = t) = probability of having exactly k-1 successes in t-1 previous time slots * probability of having a success in that last slot`
- The first part we found out how to calculate when answering the question 1, the second part is simply `p`
- This is another well-known distribution: Pascal distribution
- `E[Yk] = k/p`
- `Var(Yk) = k(1-p)/p^2`
- So, knowing probability of receiving a request each second, we can predict how many seconds would pass at the moment we receive 1000 requests
- _My thought: this sounds like period-frequency relation, does this have to be so complicated? Can't we just assume requests arriving at constant intervals, find out average interval length and multiply?_
- The process is memoryless: if you observe an interval of certain length, it does not reveal anything about the next interval
- Similarly, you can start watching the events at any point in time, provided you have chosen that moment without any foresight into the future
- Example of having "foresight into the future": looking at historical data and deciding to start the sequence based on what happened after that moment
- In essence, if you are deciding when to start watching based on what happened, it is OK, as long as you don't include those past events into the sequence

### Splitting and merging

- Imagine you have a load balancer and 2 servers
- You still have a single stream of events, but you decide which server processes the event by flipping a coin with probability `q`. The rest stays the same
- This produces 2 Bernoulli processes, one `Ber(pq)` and one `Ber(p(1-q))`
- Similarly, merging 2 Bernoulli processes is also a Bernoulli process


## Poisson process

- Memoryless, continuous time
- Continuous version of Bernoulli process
- Instead of splitting the timeline into slots, just record the timestamp of an event (basically, timeseries)
- In Bernoulli, we assumed that `p` is constant; analogous assumption in Poisson is: the number of events depends only on the length of the interval, not on the starting time
- In Bernoulli, we assumed independent trials; analogous assumption in Poisson is: the number of events in any interval is independent on number of events in any other interval
- Poisson is a default process to assume for many natural phenomena where events happen "at random": a server receiving requests at random times, particles emitted by the radioactive decay etc.
- If we take very small interval delta, so that probability of receiving more than 1 event in that interval is zero, the probability of receiving an event in that interval will be some `lambda*delta`
- This `lambda` is the "arrival rate" or "intensity of a process", and equals to expected number of events per unit of time
- Analog in Bernoulli is `p` per trial
- This essentially converts the process into a Bernoulli one
- The statement is: the Bernoulli process is a good approximation of a Poisson process, and becomes more and more accurate as delta goes to zero (and as the number of slots goes to infinity)
- So if we are interested in probability of `k` events in interval `t`, `P(k, t)`, we can take the Bernoulli PMF's and take limit as delta goes to zero, and this gives us the Poisson distribution
- If we are interested in time until `kth` event, we again follow the same approach as in Bernoulli
- The time until next event will be Exponential distribution
- The time until `kth` event will be Erlang distribution
- Practical application: to simulate Poisson process, use Exponential distribution to pick the value of the interval for the next event
- Similar to Bernoulli, merging 2 Poisson processes is also a Poisson process

### Waiting time paradox

- Imagine you have a poisson buses, and the bus company claims the expectation of interarrival time is 15'
- You come to the bus stop at random time, how long do you expect to wait for the bus?
- Well, it's poisson, so it doesn't matter how long has it been since the last bus, so it's 15' since the moment you arrive
- You might see a bus in 13' or at 17', but after series of trials, you will conclude that the average time before the arrival of the bus is indeed 15'
- At the same time, people at the bus stop tell you every time that they have been waiting for certain amount of time already before you came
- It can be 12' or 16' but on average, it will also be 15'
- Why would it be? Well, this is exactly like waiting for the first bus arrival, just with time moving backwards
- The reverse poisson is also poisson with the same parameter
- So the expected time elapsed since the last bus at the moment you arrive is indeed 15'
- Question: in that case, what is the correct interarrival time, 15' or 30'?
- Expectation says 15', but based on that, we just argued that you will observe 30'... What's wrong?
- Turns out, average interarrival time, measured (correctly) by averaging all intervals, will indeed be 15'
- It's you who are picking intervals in a biased fashion: it is much more likely for you to arrive in a middle of a large interval than in a middle of a smaller interval
- This is why you are going to observe 30' intervals on average
- Instead of picking interval at random, you should pick a bus at random and then wait until the next bus arrives
- Example of the same "paradox": average family size is 4 people, but average person comes from the family of 6 people. Both can be true, larger families are more likely to be selected when picking an average person


## Markov chains

- With memory, dependence across time
- Evolution of some variable with presence of some noise: `new state = f(old state, noise)`
- Super-useful, can be used to model many processes in a real world, for example, requests arriving to servers
- There are 3 steps to define the model
- 1) Start by identifying possible states
- The simplest version: finite discrete states on a discrete timeline
- 2) Then identify all possible transitions
- You can transition from any state to any other state with certain probability, although certain transitions may have zero probability
- 3) Finally, identify transition probabilities
- You describe the system by giving, for each state `Xi`, the probability of going to state `Xj` (does not change with time)
- Since we say "given state `Xi`", this probability is conditional on `Xi`
- **Markov property:** the probability of going to the new state `Xj` depends only on the current state `Xi` and not on any of the previous states
- That is to say, we don't care how we arrived to the state `Xi`, all that is important is that we are currently in that state
- As a result, you need to choose the state carefully: the current state needs to include everything that is relevant to produce the new state
- Example: if you predict the trajectory of a flying ball, the current position of the ball is not enough, you also need the velocity
- The goal of the model is to predict where the system will be at some future step `n` (with a certain probability)
- We could do that by considering every possible path we could take to arrive at the certain state at the given step, and calculating the probabilities of each of those paths, but that would be an extremely expensive to compute
- Thanks to Markov property, we can do that calculation recursively, one step at a time, using, on every step, the law of total probability
- Meaning all we need to know is the probability of every transition (which is given by the model definition) and the probability of being in every state at the step `n-1` (which can, in turn, be calculated in the same way)
- If you keep calculating for hundreds or thousands steps in the future, you may find out that with enough iterations, the probabilities of being in a certain state "settle" on some specific values, no matter where you start from
- This is called **steady state** of a Markov chain
- Note that this does not mean that the state transitions stop happening, it's the probabilities to be in those states that stop moving
- These probabilities can be interpreted as frequencies: how often I will be in that state
- Many Markov chains eventually enter the steady state, but not all
- Some Markov chains never enter steady state, and sometimes initial state matters (for example, if some states are not reachable from some other states)

### States

- State is **recurrent** if "there is a chance and there is a way" to get back to this state
- State is **transient** if "there are places you can go, from which you cannot come back"
- **Recurrent class** is a collection of recurrent states that "communicate" with each other (kind of clusters of recurrent states, separated by clusters of transient states)
- Recurrent class is a part of a Markov chain where you "get stuck", without being able to escape
- With multiple recurrent classes, the initial state matters, as it essentially determines in which recurrent class you are going to get stuck
- The states in a recurrent class are **periodic** if they can be grouped into `d>1` groups so that all transitions from one group lead to the next group
- The probability of state in a group will be 0 at certain times and some positive number at other times, repeating forever
- In such case you will not expect Markov chain to eventually enter the steady state, instead, the probabilities will oscillate forever
- It is not always easy to tell when a chain is periodic
- It is, however, very easy to tell when a chain is not periodic: it's enough to have a self-transition in any of the states

### Steady state convergence theorem

- The markov chain will converge to a steady state if:
- a) recurrent states are all in a single class
- b) single recurrent class is not periodic
- Given those conditions, if you take the recursive formula for calculating probability of a system to be in each of the states, then take the limit, you will get the recipe for building a system of equations (**balance equations**) that allow to find the steady state probabilities
- Those equations end up being actually quite simple (the limit goes away)
- For example, if you have state `A` and state `B`, with transitions `A->A`, `A->B`, `B->B` and `B->A`, and probabilities `P(A->A)=0.5`, `P(A->B)=0.5`, `P(B->B)=0.8` and `P(B->A)=0.2`, then the first equation looks as follows:
- `P(A) = P(A)*0.5 + P(B)*0.2`, `P(B) = P(A)*0.5 + P(B)*0.8`, which gives `0.5*P(A)=0.2*P(B)`
- To solve it, we need the second part of equation: `P(A)+P(B)=1`
- The solution is trivial
- Of course, if you have hundreds of states, solving this system of equations by hand is very tedious
- It takes certain amount of steps for the Markov chain to enter the steady state and for initial state to be forgotten, and it depends on transition probabilities. There is a whole field of study for this topic

### Absorption probabilities

- For a Markov chain that has transient states, you might want to calculate the probabilities of getting to some recurrent state/class (absorbing state), given an initial state `S1`
- You might also want to calculate number of transitions before reaching the absorbing state (expected time to absorption)
- There are ways to calculate it :D

### Application

- Example: server that receives requests, queues them and process one at the time
- Requests arrive according to `Ber(p)`, time between requests geometrically distributed, all requests are independent
- Time it takes to process a request is geometrically distributed with parameter `q`
- Arrival of requests is independent of time it takes to process them
- The probability to see the request completing depends on the state of the queue and the time it takes to process a request
- State `Xn`: number of customers in the queue
- If max size of the queue is 10, then we have 10 possible states
- An arrival of a request moves the state from `Xk` to `Xk+1`, the completion of the request moves the state from `Xk` to `Xk-1`, the simultaneous arrival and completion keeps the system at the same state
- In any of the "middle states" (queue is neither empty not full):
- the probability of the system to move to from `Xk` to `Xk+1` is `p(1-q)`
- the probability of the system to move to from `Xk` to `Xk-1` is `q(1-p)`
- the probability of the system to stay in the same state is `pq + (1-p)(1-q)`
- When the queue is empty, you cannot have any requests completing, of course
- When the queue is full, you cannot have any new arrivals
- Using this model, we could calculate the probability of queue overflow, and decide for the size of a queue
- This chain is a **death-birth process**, a special case of Markov chain
- For this kind of process, there is a clever shortcut to calculate steady state probabilities
- `r = p/q` is the **load factor**
- In a balanced process, when `r=1`, the probability of moving up is balanced by the probability of moving down, and you have an equal probability to be in any of the states (`1/(m+1)`, where `m` is the size of the queue)
- If `p<q`, the probability `P(Xn) = (1-r)*r^n`
- So `P(X0)=1-r`, `P(X1) = (1-r)*r`, etc.
- See Queuing theory for more details