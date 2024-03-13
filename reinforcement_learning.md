# Reinforcement Learning

## References

- [Stanford CS229: Machine Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- [Stanford CS234: Reinforcement Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
- [DeepMind x UCL: Reinforcement Learning](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)

## Intro

- Reinforcement learning is learning what to do (how to map situations to actions) to maximize a numerical **reward** signal
- Learning is done by interacting with the **environment**
- The learner (**the agent**) is not told which actions to take, but instead must discover which actions yield the most reward by trying them
- Actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards (delayed reward)
- One of the unique challenges is the trade-off between exploration and exploitation. The agent has to exploit what it has already experienced in order to obtain reward, but it also has to explore in order to make better action selections in the future
- We also want the agent to be able to **generalize** its experience: to learn whether some actions are good or bad in previously unseen states
- The most important component of almost all reinforcement learning algorithms is a method for efficiently estimating long-term total reward (value)


## Elements

- **Policy** is a mapping from observed states of the environment to actions to be taken when in those states
- Observed **state** can be the full state of the environment (which would make it Markov process) or just partial state (i.e. robot with camera vision, but it isn't told its absolute location, a poker player that cannot see the opponent's cards)
- To deal with partial observability, the agent can build its own state, as a function of history (an agent memory)
- Policy defines the learning agent's way of behaving at a given time
- **Deterministic policy** gives one action per state, a **Stochastic policy** picks the next action from some probability distribution
- For each interaction, an agent receives some immediate, partial feedback signal called a **Reward** (a single number)
- The agent's sole objective is to maximize a long term cumulative reward
- So it is critical that the rewards truly indicates what we want to be accomplished
- **Value function** is a prediction of future rewards under a particular policy, assigned to each state or state-action pair
- Essentially it says how good is to be in that state
- **Model** is an agent's prediction of future states and rewards based on the current state and action taken. Model serves as a simulation of a real environment
- If we don't have a model, **exploration** is required to discover actions that lead to higher rewards
- Learning from the model does not require exploration (you already have the prediction for the rewards), so it is called **planning** (aka searching)
- Methods that use models and planning are called **model-based** methods. These methods don't necessarily need policy or value function
- Methods that don't use model are called **model-free** methods. They require learning the policy and/or the value function
- The most interesting real-world applications do not have model (or it's impractical to build one)
- You may have a perfect model for the rules of the game, but no model for the opponent (model-free with respect to its opponent)
- Value-based agents do not have an explicit policy, only the value function
- Policy-based agents do not have a value function, just a policy
- **Actor critic** systems use both policy and the value function: policy is the actor, value function is the critic

### Some more intuition

- In case of a chess game, the model is a rule book for the game
- You get rewarded for winning a complete game, so you want to maximize the number of wins
- With model-based approach, you are given the complete set of rules upfront. All you need is to find the optimal sequence of moves
- With model-free approach, you start playing without the rules being explained, which leads to pure trial-and-error approach
- In this case you get a positive reward for actually winning the game, and a huge negative reward for making an illegal move
- While playing, you may still want to try figuring out the actual rules of the game (i.e. construct a model), this may help, but is not necessary
- Learning some rules as you go will allow you to avoid getting punished every time you make an illegal move
- But, instead of building the model, a policy might be all you need
- "Start game with pawn from e4 to e5" is an example of a policy. Note that you can apply this policy even without knowing game rules


## Nonassociative Problem (k-armed bandits)

- This problem avoids much of the complexity of the full reinforcement learning problem by introducing a simplified setting: there is only one situation, actions do not influence subsequent situations (nonassociative problem)
- This is essentially about exploration vs exploitation trade-off
- You are faced repeatedly with a choice among `k` different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to maximize the expected total reward over some time period, for example, over 1000 action selections, or time steps
- Analogy for this problem is a slot machine that has `k` levers instead of one, hence the name
- Action value is a mean reward given that that action is selected
- You do not know the action values with certainty, but you may have estimates
- One way to get an estimate is a **Sample-average method** which calculates an average reward from a small number of trials (the sample). The way it works is:

```
sum of rewards when a taken prior to t / number of times a taken prior to t
```

- When running algorithm on `N` time steps, you can calculate this value incrementally, updating it after every new step. The value at the step `n` is:

```
estimate_at_n+1 = estimate_at_n + (current_reward − estimate_at_n) / n
```

- Another way to look at the same formula is:

```
estimate_at_n+1 = estimate_at_n + step_size * (current_reward − estimate_at_n)
```

where step_size is `1/n`

- `step_size` can be made a hyperparameter, which can be useful later
- If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest - those are **greedy actions**
- When you select a greedy action, you are **exploiting** your current knowledge
- If instead you select one of the non-greedy actions, then you are **exploring**
- Exploitation is the right thing to do to maximize the expected reward on the one step, but exploration may produce the greater total reward in the long run
- Because it is not possible both to explore and to exploit with any single action selection, we have a conflict between exploration and exploitation
- There exist several methods for balancing exploration and exploitation
- Greedy method always exploits current knowledge to maximize immediate reward
- When the reward variances are zero, the greedy method perform best (it learns true value of each action after trying it once and then exploits)
- In the case of non-zero variances, the greedy method will most probably get locked into a suboptimal action
- Near-greedy (E-greedy) methods behave greedily most of the time, but every once in a while, with small probability `E`, select randomly from among all the actions with equal probability, independently of estimates
- In the very beginning greedy method performs best, but very quickly E-greedy methods beat the greedy one
- Higher `E` allows to find the optimal action earlier, but selects that action less times (with `E = 0.1` you are forced to spend about 10% of the time exploring, so at maximum you can exploit the best value 90% of times)
- Lower `E` improves more slowly, but eventually performs the best: not only it finds the best action, but it is also allowed to take that action more times (with `E = 0.01`, once you find the optimal action, you can exploit it 99% of the time)
- It is possible to reduce `E` over time to try to get the best of both high and low values
- If the reward probabilities change over time, it makes sense to give more weight to recent rewards than to long-past rewards
- One of the most popular ways of doing this is to use a constant `step_size` parameter, in a range `(0, 1]` (instead of `1/n` - see above)
- Having constant `step_size` have an effect of gradually moving towards the correct estimate instead of calculating precise average over all previous trials
- With `step_size` of 1, all the weight goes on the very last reward, and essentially the last reward becomes an estimate
- These methods are dependent to some extent on the initial action-value estimates: in other words, they are **biased** by their initial estimates
- For the sample-average methods, the bias disappears once all actions have been selected at least once, but for methods with constant `step_size`, the bias is permanent (we are gradually moving towards the correct estimate, but from a certain initial point)
- In practice, this kind of bias is usually not a problem and can sometimes be very helpful: it provides an easy way to supply some prior knowledge to the algorithm
- Initial action values can also be used as a simple way to encourage exploration. If we set initial values much higher than actual rewards (with constant `step_size`), then even a greedy method will be forced to switch to other actions, guaranteed to be "disappointed" with any reward it is receiving. The result is that all actions are tried several times before the value estimates converge. The system does a fair amount of exploration even if greedy actions are selected all the time
- This "optimistic initial values" method is only effective on stationary problems because its drive for exploration is inherently temporary
- Also, when exploring, instead of selecting actions randomly, we could select actions according to their potential for actually being optimal, taking into account **uncertainties** in those estimates (**Upper-Confidence-Bound Action Selection**)
- If you only selected an action once, you cannot be too certain about your estimate
- Each time action is selected, we learn more about it, so the uncertainty is presumably reduced
- Simply put, actions that have already been selected frequently, will be selected with decreasing frequency over time
- There is a mathematically proven optimal way to calculate this uncertainty and use for decision making: add bonus of `c*sqrt(log(t)/Nt(a))` to an expected reward when deciding which action to pick
- `c` is a hyperparameter, can be `sqrt(2)`, `t` is a timestep, `Nt(a)` is a number of times action `a` was selected
- When we pick a suboptimal action, we can compare the received reward with the reward that we would get from picking an optimal action, and calculate **regret**: the opportunity loss for one step
- The agent will never be able to calculate the regret, since it never knows the actual reward from the optimal action, but this can be useful for evaluating a learning algorithm
- We can set the goal for the trade-off between exploration and exploitation as to minimize the total regret (the sum of regrets over all the steps)
- Since regret can grow unbounded, we are more interested in how fast it grows
- The greedy (and E-greedy) policy has linear regret (the expected total regret), but it's possible to achieve logarithmic total regret, for example using Upper-Confidence-Bound Action Selection
- There is also a complicated proof that logarithmic total regret is the absolute best you could achieve
- **Gradient Bandit Algorithms** is another approach that relies on relative preference of one action over another
- We update action preferences in a way that maximizes the expected value of a reward, using gradient ascend
- The preference has no interpretation in terms of reward, we learn policy directly, without knowing anything about the value function
- Unlike this simplified task, in a general reinforcement learning task there is more than one situation, and the goal is to learn a **policy**: a mapping from situations to the actions that are best in those situations


## Sequential Decision Making (finite Markov decision processes)

### Definition

- **Markov Decision Processes (MDPs)** are a classical formalization of sequential decision making, where actions influence not just immediate rewards, but also subsequent situations, or states, and, through those, future rewards
- This means there is a delayed reward and the need to tradeoff immediate and delayed reward (it might be better to take smaller reward now to get into a better situation)
- Since agent always learns to maximize its reward, it is critical that the rewards we set up truly indicate what we want accomplished
- The reward signal is the way of communicating what you want to achieve, not how you want it to be achieved
- As an example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such as gaining control of the center of the board
- However, if some actions are more costly than others (e.g. consume more energy), you can model that by making a reward to be a function of both state and the action
- MDPs are a mathematically idealized form of the reinforcement learning problem for which precise theoretical statements can be made. In practice, there is a tension between breadth of applicability and mathematical tractability
- That said, the MDP framework is abstract and flexible and can be applied to many different problems in many different ways
- The learner and decision maker is called the **agent**
- Everything outside the agent it interacts with, is called the **environment**
- Anything that cannot be changed arbitrarily by the agent is considered to be outside it and thus part of its environment
- If we apply the MDP framework to a person or animal, the muscles, skeleton, and sensory organs should be considered part of the environment, not parts of the agent
- The agent-environment boundary represents the limit of the agent's absolute control, not of its knowledge: in some cases the agent may know everything about how its environment works (Rubik's cube) and still face a difficult reinforcement learning task
- In a complicated robot, many different agents may be operating at once
- We break the interactions between the agent and the environment into a sequence of discrete time steps (that can be chosen arbitrary)
- At each time step `t`, the agent receives the current state and selects an action. One time step later, the agent receives a numerical reward and a new state. The new state is a consequence of the action taken
- In a finite MDP, the sets of states, actions, and rewards all have a finite number of elements; state and reward have well defined discrete probability distributions dependent only on the preceding state and action (and they don't change over time, i.e. process is stationary)
- For every state and action taken at time step t, p is a probability of the new state and reward to happen at the next time step `t+1`
- The sum of all values of `p` for all states and rewards is naturally 1
- If `p` depends only on the immediately preceding state and action, and not on any earlier states and actions, the system is said to have a **Markov property**
- This is best viewed a restriction not on the decision process, but on the state. The state must include information about all aspects of the past agent-environment interaction that make a difference for the future
- For example, when playing chess, the current state of the board is all that matters, so it satisfies the Markov property
- But in case of hypertension control system, if we can only know the current blood pressure, it is certainly not enough to decide whether we should administer a medication or not (maybe the person has just been on the plane?). This system is not Markov
- If you can incorporate enough past history into a current state, you can make every system Markov
- MDPs assume the Markov property, but there exist approximation methods that do not rely on the Markov property

### Optimization

- We seek to maximize the **expected return** - the sequence of rewards received after time step `t`
- In the simplest case the return is the sum of the rewards. This approach makes sense in applications in which there is a natural notion of final time step or terminal state (e.g. the new chess game begins independently of how the previous one ended)
- **Episode** is a subsequence that ends in a terminal state, followed by a reset; the next episode begins independently of how the previous one ended
- Tasks with episodes of this kind are called **episodic tasks**
- In many cases the agent-environment interaction does not break naturally into identifiable episodes, but goes on continually without limit
- Tasks that go on continually without limit are called **continuing tasks**
- For continuing tasks expected return cannot be expressed as the sum of the rewards, because we cannot sum an infinite number of rewards
- To battle the infinite nature of the task, we calculate **expected discounted return**
- Basically, we don't sum the reward values directly, but multiply them by the **discount rate** `[0, 1]` in a power of `k`, where `k` is the number of time steps in the future
- This method modifies an effective value of future rewards: the more reward is in the future, the less it is worth, even if all the rest is the same
- This provides a natural cut-off for all the events that are far enough in the future
- If the reward is nonzero and constant, the sum becomes finite despite infinite number of terms (the function gets simplified to the simple expression, thanks to magic of math)
- If discount rate = 0, the agent is "myopic" in being concerned only with maximizing immediate rewards; as discount rate approaches 1, the return objective takes future rewards into account more strongly; the agent becomes more "farsighted"
- For episodic tasks, you can set discount rate to 1
- Both episodic and continuing task problem can be unified by considering episode termination to be the entering of a special **absorbing state** that transitions only to itself and that generates only rewards of zero
- Some continuing tasks may have "finite horizon". For example, when flying a helicopter, this is not an episodic task, however, you only have fuel for the next 40 minutes, so there is a natural cut-off
- In this case we don't apply the discount factor and just calculate all the way to the horizon
- Also, an optimal action in this case may depend on time (how much of it you still have left)
- A **policy** is a mapping from states to probabilities of selecting each possible action
- A policy's **value function** assign to each state, or state-action pair, the expected return from that state, or state-action pair, given that the agent uses the policy
- So, simply put, for each state, the value function tells you, how big of a reward you would get (into the unbounded future), **if you started** from that state and followed the policy
- The **optimal value functions** have the largest expected return achievable by any policy. A policy whose value functions are optimal is an optimal policy
- Policy is **stationary** if optimal action does not depend on time (the step number); otherwise, the policy is **non-stationary**

### Computing

- Value function expressed in terms of `immediate reward + sum of future rewards weighted by discount rate` can be written using recursive form, known as **Bellman equation**
- Essentially, it is formula for the expected value (see probability theory)
- The expression for the optimal value function is known as **Bellman optimality equation**
- For finite MDPs, the Bellman optimality equation for has a unique solution independent of the policy
- So in that case, in theory, we could find an optimal policy by explicitly solving the Bellman optimality equation
- In practice tasks don't fit into a strict mathematical model and optimal policies can be generated only with extreme computational cost
- Solving the Bellman optimality equation basically means "exhaustive search with infinite lookahead". It would take thousands of years on today's fastest computers to solve the Bellman equation for backgammon
- Because of that, in reinforcement learning one typically has to settle for approximate solutions
- Many reinforcement learning methods can be viewed as approximately solving the Bellman optimality equation
- Main approaches include dynamic programming (Policy iteration, Value iteration) and using samples (Monte-Carlo, Q-learning)
- Fun fact: Bellman was actually the one who developed dynamic programming methods
- Policy iteration and Value iteration rely on the perfect model (complete and accurate) of the environment as an MDP
- If you cannot rely on the model, use sampling methods instead


## Solutions

### Policy Iteration

- [MIT 16.410 Principles Of Autonomy And Decision Making, Lecture 23](https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/MIT16_410F10_lec23.pdf)
- Disclaimer: normally, less efficient comparing to Value Iteration, covered next
- Core idea: given a policy, we can always get another policy that is at least as good, by looking at the value function under that policy. This saves us the need to do exhaustive search on all policies
- Start with some (random) policy, for example, a policy where every action is equally likely
- **Policy evaluation:** given that policy, compute the value function by calculating, for each state, the expected return (by solving a Bellman equation)
- All the parameters are known: the model gives you the reward for every possible action, the (current) policy gives you the probability of every action
- You either write down a system of equations and use a linear solver library or calculate this recursively and infinitely into the future, with discount rate, until converges
- This is the trickiest part of the algorithm
- The good news is that you only have to do it for the current policy, not all the possible policies
- **Policy improvement:** based on the value function, compute the new policy
- For every state, and for every action, you look where that would bring you, and the value function tells you how good that place is. Then you update the policy in the way that brings you to the best possible next state
- So basically, you just greedily pick the best action with one step lookahead, this should produce a new policy that is at least as good as the current one
- Repeat until policy stops changing
- Policy iteration converges to the optimal policy in finite time
- This is a very bold statement, luckily, mathematicians exist to prove that all of this actually works (otherwise it might feel like you are learning "to guess from a guess")
- Problem with policy iteration is that, for large state spaces, solving the linear system for policy evaluation may be still too time-consuming

### Value Iteration

- [MIT 16.410 Principles Of Autonomy And Decision Making, Lecture 22](https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-410-principles-of-autonomy-and-decision-making-fall-2010/lecture-notes/MIT16_410F10_lec22.pdf)
- Allows to find an approximation of a value function
- Instead of looking into the future, we propagate the rewards back to the past
- For every state, initialize a value function associated with it to 0 (or any random value, except for the terminal state which should be 0)
- For every state, compute the value function at the previous step using the value function at the current step
- You do that by calculating the expected reward at the previous step + max expected value of a value function values that can be obtained at the current step, following each possible action, with a discount factor
- Repeat until converges
- So for each state, you are calculating how big of a reward you would get all the way to the final state, **if found yourself in that state k steps in the past**
- You can calculate and update the values all at once, using a second table (synchronous update) or update values one by one, sometimes looking at already updated neighbors (asynchronous update), turns out the algorithm works either way
- Value iteration converges (asymptotically) to the optimal value function monotonically and in polynomial time
- This, again, can be proven mathematically
- Simply put, you only have an approximation of a value function, but it gets better and better with every iteration
- And in case of finite horizon MDP, it gets even easier since you simply calculate all the way from the final state back, without the discount factor
- Note that you don't care about policy at all, you are searching for the optimal value function
- After you are done, the optimal policy can be easily recovered from the optimal value function
- Value iteration is used more often in practice, comparing to policy iteration

### Combined approach

- You can combine different techniques
- You don't always know the state transition probabilities. In that case it can be estimated from data
- Start with some (random) policy
- Take and action using that policy and update estimates of state transition probabilities
- Find an approximation of a value function using value iteration
- Choose the new policy that maximizes the value function
- Repeat until policy stops changing
- When doing value iteration, if you initialize value function using the values found on the previous step, this speeds up the convergence significantly
- Instead of actually computing value function, you might try to use a **function approximator** (using, for example, a deep neural network)

### Problem with DP approaches

- DP requires a complete and accurate model of the environment
- The biggest issue is the dimensionality explosion: number of states can grow really big


## Continuous states

- [B9140 Dynamic Programming & Reinforcement Learning, Lecture 5](https://djrusso.github.io/RLCourse/slides/week5.pdf)
- One of the solutions is to discretize the continuous state space
- This gives a huge state space. In practice, it only works well with 2- to 3-dimensional space
- State space is usually much larger than the action space. So in practice it's relatively easy to discretize the action space comparing to the state space
- In many real systems state is actually a function of a current state and action (as dictated by the laws of physics)
- So instead of discretizing state space, you could use a linear (or non-linear) regression to learn the function of state from the previous state and the action
- You can use a simple model like `S(t+1) = A*S(t) + B*a(t) or S(t+1) = A*S(t) + B*a(t) + e(t)` where `e` is noise (normally distributed)
- In physical robotics it is actually very important to add noise to the model (when learning)
- To get the sample data, you can use physics simulation software packages
- You can also let a human perform task and collect the samples this way
- Normally we would select some subset of state features that are enough to approximate the value function
- More data you collect, more features you can use without risk of overfitting
- **Fitted Value Iteration** is an algorithm that is based on value iteration, that learns the approximation for the value function for the continuous state space
- With Fitted Value Iteration, we use the classical statistical trick, replacing the expectation with an average
- Instead of using discrete states, you sample m states from the continuous state space. For each state and action, you sample k possible next states from the distribution of state transitions. This allows you to estimate the value function at each of m samples, using an average. After that you fit the linear regression to predict value function for all states
- Some very recent successful methods which make up "Deep Reinforcement Learning" use neural networks as function approximators

### Selecting next action in real time

- If you want to choose the best next action in real time (e.g. flying a real helicopter), you can use a simulator that predicts the next state based on the current state, for each possible actions (in this case you do not apply noise, to avoid running random number generator in real time)
- For each of next states that simulation gives you, you calculate the value function, using the previously trained model
- This gives you the best possible action you should take

### Linear quadratic regulation

- Applies when `S(t+1) = A*S(t) + B*a(t) + w(t)`, i.e. a linear function, where `w` is Gaussian noise; and the reward `R(s,a) = -(transp(s)*U*s + transp(a)*V*a)`
- `U`, `V >=0`, `transp(s)*U*s >= 0` and `transp(a)*V*a >= 0`
- Choosing `U` and `V` to be identity matrices, `R(s,a) = -(norm(s)^2 + norm(a)^2)`, this penalizes state and the action to deviate from some zero value
- With these key assumptions, you could collect some sample data and learn the matrices `A` and `B` by applying linear regression
- If the function `S(t+1) = f(s(t),a)` is non-linear, you can **linearize** it
- How this is done: at any point, the line tangent to the slope of a function can be used as an approximation of this function
- If you pick the point that your model should spend most of the time at, your approximation would apply most of the time
- Under these assumptions, the value function can be computed exactly, using DP (deriving formulas involves a lot of complicated math)
- _My understanding: this model is really good for systems that have some preferred state, and you want to keep them in that state despite some noise (like balancing the stick on the cart)_
- LQR have interesting property that, while value function depends on noise w, the optimal policy does not depend on it

### Direct policy search

- Express policy as a function of state, for example, as a sigmoid with some parameter theta (same as in logistic regression), the function telling you the probability of taking a certain action
- For example, in case of balancing the pole on the moving cart, the function will tell you with which probability you should be moving right (and if you are not moving right, you are moving left)
- The goal is to learn good theta
- This is done using gradient ascent, maximizing the expression for expected reward
- As we know, that expression is recursive, and we are looking at ever branching tree of possible next actions, but...
- ...instead of calculating over all possible branches, we are going to pick just 1 action randomly for each timestep, according to the probabilities of the current policy
- We do that to calculate the reward `t` time steps into the future, so this is `O(t)`
- If you repeat this many times (so it becomes `O(m*t)`), this will, on average, match the actual value of expected reward
- So we are going to repeat this calculation at every iteration of gradient ascent, gradually moving towards the policy that maximizes the expectation until converges
- After taking some derivatives, the update rule turns out to be quite straightforward (see the video)
- This works quite well in case of partially observable MDP (you can't measure all the parameters of your state and the ones you can measure give you noisy readings)
- The disadvantage of this algorithm is that it is very inefficient, you might need to run it for millions of iterations
- Also, it is not very good for the cases when the policy is complicated and requires planning multiple steps ahead (not very linear)


## Debugging and Diagnostics

- Setup:
- You started with building a simulator (to avoid crashing a real helicopter)
- You have chosen a reward function that matches what you want the helicopter to do
- You run RL algorithm to fly helicopter in simulation and find the policy that maximizes expected reward (with finite horizon)
- Finally, you apply the policy and find out that the resulting controller gives much worse performance than a human pilot
- What is next? Better model/simulator? Better reward function? Better algorithm?
- Does this policy do well in simulation? If yes, then the simulation is the problem
- Does the algorithm achieve better reward than a human? If not, the problem is with the algorithm
- Otherwise, the problem is in the cost function. You manage to maximize it, but it does not correspond to good autonomous flight
- If you do this diagnostics and fix an issue, you may often find out that the problem has moved (e.g. initially your simulator was not good enough, you improved it, and now you see that your cost function is not good)
