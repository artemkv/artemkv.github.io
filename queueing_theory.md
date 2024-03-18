# Queueing Theory 

* TOC
{:toc}

## References

- [MAP6264 - Queueing Theory](https://www.youtube.com/playlist?list=PL59NBu6N8dUqYClaKpoozyzK3Kpcm5eou)


## TL;DR

- If all the requests eventually get processed, offered load equals carried load
- If offered load exceeds carried load, requests get dropped/the queue grows into infinity


## Intro

- Fundamentally, this all follows form birth-death processes (see probability)
- Notation: `A`/`B`/`C`: `A` is arrival process, `B` is service process, `C` is a number of servers
- Poisson, `S` identical servers, no queue: Erlang B formula
- Poisson, `S` identical servers + infinite queue: Erlang C formula
- Non-poisson: quasirandom with or without queue


## Questions you can answer

- Average number of busy servers
- Minimal/optimal number of servers needed
- Average response time (i.e. average time a request spends in the system)
- Average queue length


## Applicability

- All models consider system in a state of statistical equilibrium
- Real-life traffic often fluctuates depending on the time of the day
- It does not mean it's not Poisson, but it definitely means that lambda may fluctuate
- In practice, you engineer your system for the busy hours, assuming that, during those hours, the things are stable and this period of time is long enough so that you could apply equilibrium results
- This way, during the busy hours the model is predicting well what happens, and during the rest of the time, you are giving a better service than one predicted by your model


## Necessary probability background

- CDF is `F(X)` and is a probability, unlike `f(x)` which is PDF
- CDF is useful to justify PDF which is not intuitive, as it does not express probability
- Computer random number generator has uniform distribution (but can be used to produce values from other distributions, using inverse transform method, see below)
- `E[X]` is the "first moment"
- `E[X^2]` is the "second moment"
- Poisson is the default process to assume, unless there is a good reason not to
- Exponential distribution is time between arrivals, when events happen "at random" (Poisson) with some arrival rate lambda
- `E[X] = 1/lambda`
- `Var(X) = 1/lambda^2`
- Magically, if you waited time `y` and there was no arrival, the probability to have an event in the next interval `y+t` does not depend on `y` ("Markov memoriless property")
- Probability of having `j` arrivals in time interval `t` is a Poisson distribution


## Taxis and customers

- `E[X]`: expected length of an interval between 2 taxis
- `E[I]`: expected length of an interval in which a customer arrives (interrupted interval)
- Turns out, `E[I] = E[X] + Var(X)/E[X]`
- So, somewhat counter-intuitively at first, `E[I]` is (generally) larger than `E[X]`
- Explanation: imagine `P(X=1) = 9/10`, `P(X=11) = 1/10`, so `E[X] = 2`. Basically, if you gamble on this, you would, on average, win 2 on every 10 tosses
- Those 10 tosses would, on average, last 20 units of length, 9 units being taken by 1-length intervals, 11 units by 11-length interval
- So the fraction of time the arriving request would land on a shorter interval is 9/20, i.e. `P(I=1) = 9/20` (same `P(I=11) = 11/20`)
- This is actually not that surprising if you think that the larger the interval, the more chances there is for the customer to land in that interval
- `E[R]`: expected time the customer has to wait for the next taxi
- `E[R] = 0.5*E[I]`, as we can land anywhere in that interval
- Looking at the formula it's easy to see that, depending on values of `Var(X)` and `E[X]`, you may be able to make `E[R]` smaller by making `E[X]` larger
- In other words, you can make the waiting times shorter by making the taxis arrive at longer intervals
- In our example, `Var(X) = 9`, so `E[R] = 6.5` when `E[X] = 2` and `E[R] = 6` when `E[X] = 3`
- However, `E[R] = 10.9` when `E[X] = 10`, so you cannot just increase `E[X]` forever :)
- This is why, in manufacturing and other domains, sometimes introducing forced idle time into the system makes it more efficient


## Inverse transform method

- This is the method for using computer random number generator (having uniform distribution) to generate random variable values from any specified distribution
- Algorithm: given any arbitrary CDF `G(U)`, pick a random number `R`, put on a vertical axis, project to the curve `G(U)` and down to the horizontal axis
- Generate `R` using random number generator
- CDF has a nice property: all `G(U)` fall into the range `[0, 1]`, and it never oscillates
- To make the projection, you need to find an inverse of `G(U)`, hence the name
- It can be proven that given `R` is uniformly distributed, the new, transformed values are going to follow the distribution `G(U)`


## Simulation

- This simulation is for `S` servers, no queue

```basic
100 DIM C(50)                       ' server availability times
110 INPUT S, NSTOP                  ' S is number of servers, NSTOP number of requests to simulate
120 FOR D = 1 TO NSTOP              ' run for NSTOP iterations
130 IA = ???                        ' IA is interarrival time, generate from distribution
140 A = A + IA                      ' A is the arrival time of the previous customer, acts as the clock
150 J = 0                           ' J is the index of a server
160 J = J + 1
170 IF J = S + 1 THEN K = K + 1     ' K is number of overflowed requests
180 IF J = S + 1 THEN 270
190 IF A < C(J) THEN 160            ' look for a server to put the request to
200 X = ???                         ' X is the service time, generate from distribution
210 C(J) = A + X                    ' A + X is the time the server is going to become available
220 M = C(1)                        ' M is the time next server becomes available
230 FOR I = 2 TO S                  ' we are going to find it by iterating over all servers
240 IF C(I) < M THEN M = C(I)
250 NEXT I
260 IF M > A THEN AB = AB + M - A   ' if all busy, remember AB, the time all servers are busy
270 NEXT D
280 PRINT K/NSTOP                   ' probability of dropping a request
280 PRINT AB/A                      ' fraction of time we are in the blocking state
```

- If we add the queue, the difference is, when all servers are busy, we don't drop requests
- Instead, we find the first server that will become available
- We schedule the request on that server
- Waiting time is the delta between the time request arrived and the time the first server will become available
- There is no need to do anything special for the queue, actually


## Erlang B formula

- Requests arriving to `S` identical servers, no queue
- **Input process** is Poisson (important!)
- Arrival rate `lambda` (number of requests per unit of time)
- Average service time `tau`
- **Service process**: when no servers available, the request is dropped
- You want to reserve as few servers as possible to save on resources, but enough not to drop too many requests
- You could, of course, provision for the average, but this would not handle spikes in traffic well
- You want to be able to calculate how many servers you need to guarantee the certain probability of serving a request
- `N(t)` is the number of requests that are being processed at time `t` (number of servers that are busy)
- State `n` correspond to one of the possible values of `N(t)`, which is `0..S`, so every state is described by the number of requests being processed (or the number of servers that are busy)
- `Pn` is the probability of being in state `n`
- `Pn` is calculated as time the system spends in state `n` to the total time, as observed by the 3rd party observer
- This system is a Markov chain, and it fulfills the conditions for the steady state
- However, at the moment Erlang studied this, Markov chains were not described
- Clever insight: in the long run, for every `N(t)=n`, the rate of going up (the state) = rate of going down (the state) (i.e. eventually every request gets processed): "conservation of flow"
- This allows us to write series of balance equations (expressing probabilities for the steady state of a Markov chain):
- `lambda*P0 = (1/tau)*P1` (average rate requests arrive, and we are in `P0` = average rate requests complete, and we are in `P1`)
- `lambda*P1 = (2/tau)*P1` (with 2 requests in the queue, the probability of 1 of them completing is twice as high)
- ...
- `lambda*P(S-1) = (S/tau)*PS`
- Note: by bringing up the "conservation of flow", we appeal to intuition, however, if you really think about it, it is not intuitive at all. Later we'll see that all of this can be formally derived through math (see Birth and death processes), and it can also can be proven by simulation (TODO)
- The total probability (`P0 + P1 + ... + PS`) must be equal to 1, which allows completing and solving this system of equations
- `PS` has a special meaning, since, essentially, it is the probability of losing a request (due to all servers being busy)
- If you express `PS` through `lambda`, `tau` and `S`, you get the famous "Erlang B formula" (or "Erlang loss formula" or "Erlang first formula")
- Turns out, `PS` only depends on product of `lambda` and `tau` (not their individual values), so we introduce `a = lambda*tau`
- `a` is called **offered load**, measured in erlangs (think of it as a measure of demand on the system)
- Erlang B formula `B(S,a) = PS` expresses the probability of a request loss for a group of identical parallel resources, as a function of `S`, `a`
- The formula provides the grade of service which is the probability `Pb` that a new request arriving to the resources group is rejected because all resources (servers, lines, circuits) are busy
- The actual formula itself is somewhat complicated, so look it up (And see Erlang loss model for a shortcut formula)
- Normally, you would pick the reasonable `Pb`, and calculate the amount of servers required to guarantee this number
- This is what phone companies used to do to calculate number of trunks to provide
- This is, of course, assuming you know `lambda` and `tau`
- Estimating the `lambda` and `tau` is the domain of statistics, so it goes out of scope of this course, which is focused on probability
- Amazingly, this analysis does not care about the `tau` distribution
- Note that `Pn` is, in general, different from `Pin`, which is the probability that the arriving request will find the system in state `n`
- However, if input process is Poisson, `Pn == Pin` (See PASTA theorem)
- We actually used this property for our "clever insight", which, in fact, should be written as `lambda*Pi(S-1) = (S/tau)*PS` (`Pi` on the left side and `P` on the right one)


## Erlang C formula

- Requests arriving to `S` identical servers + infinite queue
- **Input process** is Poisson (important!)
- Arrival rate `lambda`
- Average service time `tau`, exponentially distributed (unlike in case with no queue) (important!)
- **Service process**: when no servers available, the request is queued, those requests stay in the system until they can be handled
- So no request is dropped, but they can be delayed (possibly infinitely)
- We are going to use the same argument: "what goes up must come down"
- We start by writing down the same equations as before, for states `P0...PS`:
- `lambda*P0 = (1/tau)*P1`
- ...
- `lambda*P(S-1) = (S/tau)*PS`
- State `S`: all servers are busy, queue is empty
- States `S+1, S+2, ... S+infinity`: all the servers are busy, queue contains at least one item
- In all equations for states `≥ S`, the enumerator will always stay `S`, because with `S` servers, there is max `S` requests that can get completed:
- `lambda*PS = (S/tau)*P(S+1)`
- `lambda*P(S+1) = (S/tau)*P(S+2)`
- ...
- `lambda*P(S+k) = (S/tau)*P(S+k+1)`
- ... into infinity
- As always, the total probability must be equal to 1, which, same as before, allows completing and solving this system of equations
- The part that goes into infinity converges under `a/S < 1`, i.e. `lambda < S/tau`
- As before, `a` is **offered load** = `lambda*tau` (think of it as a measure of demand)
- So essentially, in the long run, the servers should be able to process requests faster than they arrive, otherwise the queue will grow into infinity
- We are interested in `C(S,a)`, the probability of delay, i.e. the probability of request being queued ("Erlang C formula" or "Erlang second formula")
- It's easy to see that `C(S,a) = PS + P(S+1) + P(S+2) + ...`
- The actual formula itself is somewhat complicated, so look it up (And see Erlang delay model for a shortcut formula)
- Unlike for B formula, `tau` is required to follow exponential distribution
- Queue order of service does not matter, as long as the customer selected from the queue is statistically identical to any other customer
- So LIFO, FIFO, random selection etc. is fine, but not the "shortest time first"
- You can use this formula to calculate probability of being in state `S+j`
- With only 1 server, `C(1,a) = a`


## S identical servers + finite queue

- Requests arriving to `S` identical servers + finite queue of size `n`
- **Input process** is Poisson
- Arrival rate `lambda`
- Average service time `tau`, exponentially distributed (unlike in case with no queue)
- **Service process**: when no servers available, the request is queued, if queue is full, the request is dropped
- So now we are dealing with both request drops and delays
- The same equations we used for C formula still work, with only difference that we don't need to go into the infinity:
- `lambda*P(j) = ((j+1)/tau)*P(j+1), for all j = 0 ... S-1`
- `lambda*P(j) = (S/tau)*P(j+1), for all j = S ... S+n-1`
- We don't have to look at `P(S+n)`, since the rate of going up from `S+n` is 0, same as `P(S+n+1)`
- Following the exact same approach as before, we solve this system of equations and find expressions for all `P`
- In fact, it gets better, because, when we had an infinite queue, we required `a/S < 1` for the expression to converge, with the finite queue, this requirement is dropped
- We can use these expressions to find the probability of a request being queued `(PS + P(S+1) + P(S+2) + ... + P(S+n-1))` or being dropped `(P(S+n))`


## Birth and death processes

- This is formal math model, from where our "intuitive" model discussed above can be derived from (so it replaces a handwoven argument of conservation of flow)
- _My note: this can be skipped for all practical purposes, but it is the only thing you need if you want to derive everything from scratch_
- `N(t)` is a population size at time `t`
- When a birth occur, `N(t)` goes up by 1, when a death occur, `N(t)` goes down by 1
- We look at probability `P(N(t+h)=j)`, which, by the law of total probability can be expressed as `[sum of (P(N(t+h)=j|N(t)=i)*P(N(t)=i)) for all i]`
- `i` can be `j-1`, `j+1` or `j`, and possibly all other values, but with sufficiently small `h`, we can discard all other values of `i` (so `N(t)` can only go up or down 1 step at a time)
- That means we can express `P(N(t+h)=j)` through just `P(N(t)=j-1)`, `P(N(t)=j+1)` and `P(N(t)=j)`
- We also need to know transition probabilities, i.e. `P(N(t+h)=j|N(t)=i)` for these 3 possible values of i
- By modeling assumption, the transition probabilities will be proportional to the length of interval `h` through some constants (plus some `O(h)` that will disappear when we take a limit, so I omit it here)
- We let `P(N(t+h)=j|N(t)=j-1) = lambda(j-1)*h`, where `lambda` is "birth coefficient" and depends on population size
- We let `P(N(t+h)=j|N(t)=j+1) = mu(j+1)*h`, where `mu` is "death coefficient" and depends on population size
- We let `P(N(t+h)=j|N(t)=j) = [1 - (lambda(j-1)*h + mu(j+1)*h)]` = "no birth neither death" (we ignore the case of simultaneous birth and death, since with `h` small enough this can't happen)
- Importantly, the transition probabilities do not depend on `t`!!! Of course, since lambda and mu depend on population size, they indirectly depend on `t`, but this is implicit, and we don't have to care about it
- Basically, this means `lambda` and `mu` capture all the state we need, regardless of how we got there (Markov property)
- Turns out, these modeling assumptions don't have to be pulled out of thin air, but can be derived from the requirement of interarrival interval and service time being exponentially distributed (see "Exponential distribution")
- We take a limit with `h → 0`, and convert the whole thing into a differential equation
- Now we can consider some specific cases of this process: pure birth process, pure death process and alternating renewal process
- With **Pure birth process**, when there are no deaths (`mu=0`) and `lambda` is constant for any population size, this equation magically converts into Poisson distribution with expectation `E[N(t)] = lambda*t`
- Due to the constant `lambda`, this process is not well-suited for modeling a biological population, but it could be good for modeling message arrivals
- With **Pure death process**, when there are no birth (`lambda=0`) and `mu(j) = j*mu`, i.e. proportional to the population size, this equation magically converts into Binomial distribution, where `p` is has a very specific form
- That is, if all the individual members of a population had independent, exponentially distributed lifetimes, `p` would be the probability, for each of them, to still be alive at time `t`
- If we combine birth and death processes, the equation becomes way too difficult to solve explicitly, so we look at the particular case of it, alternating renewal process
- With **alternating renewal process**, a system alternates, over time, between 0 and 1 (this would be analogous to 1 server and no queue)
- We let `lambda(0) = lambda` and `lambda(1) = 0` (only allow births when in state 0), `mu(1) = mu` and `mu(0) = 0` (only allow deaths when in state 1)
- Plugging these value into the equation, we will get expressions for `P(N(t)=0)` and `P(N(t)=1)`
- If we plot those, we'll see that `P(N(t)=1)` will approach, asymptotically, `lambda/(lambda+mu)`, and `P(N(t)=0)` will approach, asymptotically, `mu/(lambda+mu)`
- If we look at the system in **statistical equilibrium**, by taking a limit with t → infinity, those probabilities become **stationary**, i.e. `P(N(t)=1) = lambda/(lambda+mu)` and `P(N(t)=0) = mu/(lambda+mu)`
- There are 2 ways to interpret the `P(N(t)=1) = lambda/(lambda+mu)`
- On one side, `lambda/(lambda+mu)` is a fraction of time a single system is in state 1
- On the other hand, `lambda/(lambda+mu)` is a fraction of systems that are in the state 1, looking at infinite number of (identical) systems at any point in time
- Since, in statistical equilibrium both `P(N(t)=1)` and `P(N(t)=0)` are going to be stationary, their derivative will be 0, this allows us plugging 0 back into the differential equation
- And then, magically, we get `lambda(0)*P0 = mu(1)*P1`, the same exact thing we assumed when we said "rate up equals rate down"
- If we assume equilibrium condition for birth and death process in general, we can follow the same process and derive all the rest of the expressions for states >1
- Turns out, you will arrive at the general expression for "rate up equals rate down":

```
lambda(j)*P(j) = mu(j+1)*P(j+1)
```

- This gives perfect mathematical justification for the previously discussed models, removing the need for a handwavy argument "rate up equals rate down"
- And this is why we need a strict assumption about process being Poisson and the tau be exponentially distributed
- And actually, it gets even more justified and more formal when looking at exponential distribution properties

### Exponential distribution

- _My note: this, in a way, is even more fundamental that the birth-death process_
- Examples: lifetime, interarrival time
- Exponential distribution has an interesting property that the minimum of many independent exponentially distributed random variables (with parameters `lambda1`, `lambda2`, ..., `lambdaN`) is also exponentially distributed, with parameter (`lambda1 + lambda2 + ... + lambdaN`)
- So given many independent exponentially distributed lifetimes, the time until the first death is also exponentially distributed, and the parameter of this distribution is the sum of the parameters of each individual lifetime
- Same, with multiple independent sources generating messages at exponentially distributed intervals, the time until the first arrival to our system is also exponentially distributed
- Same, if arrivals and lifetimes are both exponentially distributed, and a step-up / step down happens depending on which event happens first, the time between those events (of going up or down) is also exponentially distributed
- You can think about it as a race
- When 2 independent exponentially distributed random variables `X1` and `X2`, probability that `X1` "wins the race", `P(X1 < X2) = lambda1 / (lambda1 + lambda2)`
- Under same conditions, `P(min(X1, X2)>t | X1<X2) = P(min(X1, X2)>t)`, so time to win the race is independent of who won the race
- **The whole point of this:** turns out, if you require arrivals and lifetimes to be both exponentially distributed, you can actually derive the modeling assumptions that we made about the transition probabilities
- Meaning you can derive everything that was discussed above from just this assumption


## PASTA theorem (Poisson Arrivals See Time Averages)

- `P(N(t)=j)` is the probability of a system to be in the state `j`
- It is a fraction of time system is in the state `j`, as observed by an outside observer
- `P(N(t)=j|arrival in t,t+h)`, as `h → 0`, is the probability of an arriving request to find system is the state `j`
- The theorem states that, for Poisson arrivals, magically these 2 probabilities are the same (the proof is not very difficult, but not important)


## Offered load, carried load and utilization

- **Offered load** `a = lambda*tau`, measured in erlangs
- **Carried load** `a'` is an average number of busy servers in a group, can be `S` at maximum, also measured in erlangs
- Carried load expresses the amount of work done
- `'a = 0*P(0) + 1*P(1) + ... + (S-1)*P(S-1) + S*P(S) + S*P(S+1) + ...`
- Carried load on a single server is just a probability of that server being busy
- This has a form of expectation (for an indicator r.v.), and by linearity of expectations, the loads of multiple servers simply add up
- This means, if you are running a simulation (with Poisson), `a'` can be calculated as `[sum of the service times of all carried requests]/[total time simulated]` TODO: prove by simulating
- _My note: this is a beautiful duality between fraction of time the servers are busy and the number of servers being busy_
- **Utilization** `ro` is `a'/S` (100% when all the servers are busy all the time)


## Little's theorem

- `L = lambda'*W`: expected queue length is the effective arrival rate times expected waiting time (regardless of distribution, in a stationary system)
- **Arrival rate** `lambda`: rate at which customers arrive at the store
- **Effective arrival rate** `lambda'`: rate at which customers enter the store
- In a system with an infinite size and no loss, the two are equal


## Erlang loss model (M/G/s/s)

- M/G/s/s = Markov/General(no need to be exponential)/`s` servers/no queue
- _My note: this is revisiting B formula_
- Poisson arrivals, `lambda` is constant
- Exponential service time, `mu(j) = j*mu = j*(1/tau)`
- Rate up = rate down
- Since there is no queue, `'a = 0*P(0) + 1*P(1) + ... + (S-1)*P(S-1)`
- From this, it can be shown that `a' = a*(1-B(S,a)) = a - a*B(S,a)`
- Meaning carried load is a fraction of an offered load that is "carried" (i.e. not dropped)
- If all the requests eventually get processed, offered load equals carried load
- Re-arranging the previous expression, `B(S,a) = (a-a')/a = 1 - a'/a`
- So if offered load exceeds carried load, requests get dropped
- So the easy way to avoid requests to be dropped is to simply provision `a` servers, but that might be an overkill (too expensive)
- How do you calculate a minimal number of servers to guarantee `Pb = B(S,a)` is less than a certain number? You can use Erlang B formula directly, but that is complicated
- Turns out, `B(S,a)` can be calculated recursively, using `B(S,a) = (a*B(S-1,a))/(S + a*B(S-1,a))`
- So you can start with `S=1` and loop until you get the `B(S,a)` that you are satisfied with

```
B(S,a) = (a*B(S-1,a))/(S + a*B(S-1,a))
```

- If you plot `B(S,a)` as a function of `a`, with a given number of servers `S`, you will get a (logarithmic) curve, asymptotically approaching 1
- `a` is determined by the external demand, `B(S,a)` is decided by the system requirements, so as an engineer, you have to come up with the number `S` (how many servers to provide to guarantee the desired grade of service)
- Your goal is to find the number of servers `S`, that produces the curve that passes the closest to the point `[a; B(S,a)]` and lays below it
- This means adding just 1 more server would move the curve above that point
- One thing you might notice that, if you split your traffic between 2 server groups, the total number of servers required will be bigger than in case of having just one group of servers (e.g. 117 for 100 erlangs with `Pb=0.01`, but when split into 2 groups 50 erlangs each, each group would require 64 servers)
- To apply Little's formula, `L` is `a'`, `W` is `tau` and `lambda'` is `lambda*(1-B(S,a))`, i.e. request that enter the system are the ones that are not dropped

```
a' = lambda*(1-B(S,a))*tau = a*(1-B(S,a))
```

### High and low priority traffic exercise for Erlang B

- **Setup:** 2 streams of requests arrive at a server group, 1st stream is a high-priority stream, `a_h` erlangs; 2nd stream is a low-priority stream, `a_l` erlangs
- Both streams are handled by the primary group of S=10 servers
- Low priority requests that overflow the primary group are dropped immediately
- High priority requests that overflow the primary group are routed to a backup of C overflow servers
- Both streams are Poisson
- A measure shows that the requests overflow at a rate of 2 per hour
- `lambda_h` = 20 per hour, `tau_h` = 12' = 1/5 hour
- `lambda_l` and `tau_l` are unknown
- **Solution:** given this, `a_h = 20*(1/5)` = 4 erlangs
- (TODO:) turns out, when there is no queue, you can add erlangs, so the total offered load is `a_h + a_l`
- So the probability for a request to overflow will be `B(S, a_h + a_l)`
- Claim: By PASTA, both streams, being Poisson, will have the same probability of finding a system to be in state `S`, which will be `PS = B(S, a_h + a_l)`, which means the same fraction of requests will be dropped
- Based on that, next claim: `B(S, a_h + a_l)` can be calculated as 2/20
- _My note: I don't really buy this, it seems to me that it should be `2/(20 + lambda_l)`_
- Knowing `B(S, a_h + a_l)` and `S`, then, of course, you can calculate `a_h + a_l` to be 7.5, so `a_l` = 3.5
- _My note: using this result, valid values for `lambda_l` and `tau_l` could be 35 and 1/10 respectively. But then, it seems, it should be legit to calculate `B(S, a_h + a_l)` as 2/35, following the same logic. But that gives completely different answer. So I'm missing something, because it doesn't make sense_
- Turns out, the overflow traffic routed to a backup of `C` overflow servers is not Poisson, so the erlang B formula can't be applied (which is natural if you think the overflow happens during peaks of load)


## Erlang B ordered hunt

- System with potentially infinite number of servers `1, 2, 3, ... j-1, j`
- To serve incoming requests, servers are selected in order, i.e. if the server 1 is free, it processes the request, otherwise the request goes to the server 2, and so on
- The question we are trying to solve is `a_j`, carried load to server `j`
- Since this is the only server (a group of 1 server), `a_j` is the same as just the probability of server `j` being busy
- With `Xj=0` when idle, `Xj=1` when busy, `a_j = E[Xj] = P(Xj=1)`
- Computing this probability directly is a difficult problem, but we could use some computational shortcuts
- Let `N(j)` be the number of servers busy among first `j` servers
- `N(j) = X1 + X2 + ... + X(j-1) + Xj`
- ⇒ `E[N(j)] = E[X1 + X2 + ... + X(j-1)] + E[Xj]`, given `X1, X2 ...` are independent
- ⇒ `E[N(j)] = E[N(j-1)] + P(Xj=1)`
- ⇒ `P(Xj=1) = E[N(j)] - E[N(j-1)]`, rearranging the expression
- then, using `a' = a*(1-B(S,a))`,
- ⇒ `P(Xj=1) = a*(1-B(j,a)) - a*(1-B(j-1,a)) = a*B(j-1,a) - a*B(j,a)`
- so, `a_j = a*B(j-1,a) - a*B(j,a)`
- And only the offered load on server 1 has to be Poisson (we know that overflowed load is not Poisson)


## Erlang delay model (M/M/s)

- M/M/s = Markov(Poisson)/Markov(Exponential)/`s` servers
- _My note: this is revisiting C formula_
- Poisson arrivals, `lambda` is constant
- Exponential service time, `mu(j) = j*mu` when `j < S`; `mu(j) = S*mu` when `j >= S`
- Rate up = rate down
- If you plot `C(S,a)` as a function of `a`, with a given number of servers `S`, you will get a (logarithmic) curve, going from 0 to 1 (on vertical axis)
- Since every request gets carried eventually (thanks to infinite queue), `a' = a`
- So utilization `ro = a'/S` is equal to `a/S` for `a < S` (and is equal to 1 for `a >= S`)
- Turns out, `C(S,a)` can be calculated from `B(S,a)`, so you can do it iteratively as well

```
C(S,a) = S*B(S,a)/(S-a*(1-B(S,a)))
```

- Using our previous definition of `ro`, `C(S,a) = B(S,a)/(1-ro*(1-B(S,a)))`
- For given `S` and `a`, `C(S,a) > B(S,a)`

### Waiting time

- One of the important concerns is waiting time distribution (meaning waiting in the queue)
- Assume FIFO queue (important!)
- Formally, we are interested in `P(W>t)`, the probability that requests waits > time `t` in the queue (`W` is waiting time)
- Using the classical trick, `P(W>t) = [sum of Pi(N=k)*P(W>t|N=k) for all k >= S]`
- Remember, for Poisson, `P(N=k) = Pi(N=k)`, so
- `P(W>t) = [sum of P(N=S+j)*P(W>t|N=S+j) for all j >= 0]`
- Since, for the request to advance through the queue, requests need to complete, we can also express `P(W>t)` in terms of `P(Dep(t)=i)`, the probability of having exactly `i` departures within time `t`
- Time until first departure is exponential (see Exponential distribution) with rate `S*mu`
- So, as long as all servers are busy, the departure process is Poisson, which allows deriving `P(Dep(t)=i)`
- Combining these 2, and doing some arithmetic,
- `P(W>t) = C(S,a)*e^(-(1-ro)*S*mu*t)`
- This also means `P(W>t|W>0) = e^(-(1-ro)*S*mu*t)`, meaning exponentially distributed (`W>0` means "given you have to wait")
- `E[W|W>0] = 1/((1-ro)*S)`
- `E[W] = (C(S,a)*tau)/((1-ro)*S)`

```
E[W] = (C(S,a)*tau)/((1-ro)*S)
```

- `Fw(t) = 0` when `t < 0`; otherwise `1-P(W>t)`
- Using Little's theorem, expected queue length `E[Q]=lambda*E[W]`
- Since there is an infinite queue, all requests enter the system, so `lambda'=lambda`
- From that, `E[Q] = C(S,a) * (ro/(1-ro))`

```
E[Q] = C(S,a) * (ro/(1-ro))
```
- Expected number of requests in the system `E[N] = E[Q] + a'`, expected number of requests in the queue + average number of busy servers
- Since `a' = a`, `E[N] = C(S,a) * (ro/(1-ro)) + a`

```
E[N] = C(S,a) * (ro/(1-ro)) + a
```

- Expected response time `E[R] = E[W] + tau`, expected time to wait in the queue + average service time

```
E[R] = E[W] + tau
```

### M/M/1

- Special case of M/M/s
- `C(1,a) = a`
- `a = a' = ro`
- `E[Q] = ro^2/(1-ro)`
- `E[W] = (ro/(1-ro))*tau`
- `E[N] = ro/(1-ro)`
- `E[R] = (1/(1-ro))*tau`


## Finite source models aka Quasirandom input (quasi-Poisson)

- Finite number `n` of sources that request service (from time to time)
- While source is making request, it cannot make another request, which makes request dependent (and not Poisson)
- In case of telecommunication, this corresponds to lines that connect to trunks, could be used for the Internet traffic
- In the old days, this was used to model terminals connecting to a shared CPU
- Climbing gym analogy: `n` climbers in the gym that have access to `S` boulder problems, and periodically want to climb those (assuming no new climbers arrive and none leave)
- Since none of it is Poisson, it is claimed to be a good model for those cases
- People use it to model Internet traffic
- Service is considered to be in "On" state when served; otherwise "Off"
- If there is an available server, the source stays "On" while served, then goes back "Off"
- So this is an alternating renewal process
- For each source, the time until it makes a request ("think time") is exponentially distributed with parameter `gamma` (although it will be shown not to depend on distribution)
- Average think time is `1/gamma` (see mean of exponential distribution)
- Average service time `tau`, exponentially distributed (same, will be shown not to depend on distribution)
- The probability to get exactly `j` requests for service in the interval of length `x` is Binomial distribution
- Taking a limit with `n → infinity`, while maintaining the same traffic (i.e. rate per source → 0), Binomial distribution becomes Poisson, i.e. this input process converges to Poisson
- It can be shown that `Pi[n,j] = P[n-1,j]`, i.e. in a system with `n` sources, the probability of the arriving request to find the system in state `j` equals probability of a system with `n-1` source to be in state `j`
- This work for any queue policy (whether there is a queue or not)
- And, the most importantly, this, apparently, holds regardless of distribution of think time or service time, and only depends on their averages

### 1 source 1 server

- Server goes up and down in cycles, every cycle starts with think time (`1/gamma`) and follows with service time (`tau`)
- Calculating the probability of a server being busy is just counting the fraction of time it is busy
- Which is `P[1,1] = tau/(1/gamma + tau) = (gamma*tau)/(1+gamma*tau)`
- Let's call `a_hat = gamma*tau` an offered load per idle source
- This gives `P[1,1] = a_hat/(1+a_hat)`
- One source can never find the service busy, so `Pi[1,1] = 0`
- Intuitively, it's easy to see that none of it depends on distributions (we are just looking at cycles, and they only depend on averages of `gamma` and `tau`)

### 2 sources 1 server

- Blocked customers cleared
- The time to the first request is a minimum of think times of 2 sources
- As we have seen before, the min of 2 exponential distributions with `gamma` is an exponential distribution with `2*gamma`
- The service time is just `tau`, like before
- So `P[2,1] = tau/(1/(2*gamma) + tau) = ... = (2*a_hat)/(1+2*a_hat)`
- `Pi[2,1]` turns out to be `a_hat/(1+a_hat)`, which confirms the previous statement that `Pi[2,1]` should equal `P[1,1]`
- Why so? While server is busy with one source, the amount of requests the second one can make is proportional to the service time and `gamma`, i.e. = `gamma*tau`
- The amount of requests that is made during non-busy time is exactly 1 (the one that makes it busy)
- `Pi[2,1]` is a fraction of requests that find server busy to the total amount of requests, and if you do that, the math works out
- At this point it is still intuitively clear the service time distribution is not important, however, the same cannot be said about the think time (since we had to make an assumption about it being exponential), but it is still true nevertheless

### n sources, S servers, blocked customers cleared

- This model specifically is claimed to be good for Internet traffic
- This is very similar to Erlang B formula, except `lambda` is not constant anymore and depends on the state of the system, making it not Poisson
- `tau` is still constant
- But we are not going to use B formula, and instead look at it as a birth-death process: `lambda(j)*P(j) = mu(j+1)*P(j+1)`
- `lambda(j) = (n - j)*gamma`
- Explanation: `lambda(j)` is a value of lambda when `j` servers are busy; `(n - j)` is the number of sources that are not being served (and hence may request the service)
- `mu(j+1) = (j+1)/tau`
- Explanation: with `j+1` servers are busy, and each processing the task, taking service time `tau`, any of them can free up in `1/tau`
- We also know that offered load per idle source is `a_hat = gamma*tau`
- This allows to find state probabilities `P[n,j]` in terms of `a_hat`
- `P[n,j] = [(n choose j)*a_hat^j]/[sum of [(n choose k)*a_hat^k], for all k = 0 ... S]`
- `Pi[n,S]` is a fraction of customers that will find system in the state `S`, i.e. that will be blocked, and it was said above to be equal to `P[n-1,S]`, so can be calculated from the formula above
- Instead of using this formula, which is unwieldy, you can follow a numerical procedure: take `P(0) = 1`, calculate `P(1)` from it, and so on; at the end don't forget to normalize by the sum, so that all the probabilities add up to 1
- All you need to know is the average think time and the average service time of a source
- Intended offered load `a_star = n * (a_hat/(1 + a_hat))`
- Carried load `a' = a_star*[1-(1-S/n)*P[n,S]]`

### n sources, S servers, blocked customers delayed

- This is very similar to Erlang C formula, except `lambda` is not constant anymore and depends on the state of the system, making it not Poisson
- `tau` is still constant
- Again, we use birth-death process: `lambda(j)*P(j) = mu(j+1)*P(j+1)`
- `lambda(j) = (n - j)*gamma`, same as above, with blocked customers cleared
- `mu(j) = j*mu` when `j < S`; `mu(j) = S*mu` when `j >= S`, same as when discussing delay model
- This allows to find state probabilities `P[n,j]`, which turns out to be such a huge formula, that I won't even try to type it here
- You can also look at `P(W>t)`, but that's also a hugely complicated formula
- Same for `E[W]`
- The good news is, all you basically need is still average think time and the average service time

#### Applying Little's theorem

- We can look at it as a close system: `n` sources, `S` servers, if all servers are busy, requests get queued; once served, source leaves the system and immediately re-enters it
- Let's try applying Little's theorem to it (`L=lambda'*W`)
- `L` is `n`
- `lambda'` is `lambda`, since no requests get lost
- `W` is think time `1/gamma` + wait time `W` + service time `tau`
- So `n = lambda(1/gamma + W + tau)`
- Since the rate the requests are processed equals the rate the requests are generated, `lambda` is a throughput of such system
- So this formula can be used to calculate the throughput, which can be useful for computer performance evaluation