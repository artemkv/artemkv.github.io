## Search

- **Reflex-based model:** given `x`, output an action `y`
- **Search problem:** given state `s1`, possible actions `A(s)`, cost of every action `C(s, a)`, next state `Succ(s, a)`, find a sequence of actions `a1, a2, ... an` that leads from `s1` to `s2` with the least cost
- __My thought__: Everything in life can be framed as a search problem
- **Learning problem** can be seen as an inverse of search problem: given the optimal sequence `a1, a2, ... an` find the cost `Cost(s, a)`
- 3 approaches: tree search, DP and Uniform cost search
- Tree search: any time you have a choice of action to take, branch
- **Backtracking search**: explore every possible path, depth first
- **DFS**: same as backtracking search, but we stop once we find the first possible solution
- The solution may not be on the shortest path, but the search is very memory efficient
- **BFS**: explore all branches layer by layer, stop once we find the first possible solution
- Comparing to DFS, will find the solution closest to the root. However, in terms of space, BFS is actually quite much worse than DFS. TODO: actually elaborate on that
- **DFS-ID** (DFS - iterative deepening): set max depth, perform DFS. If found, return. If didn't, increase max depth, repeat (from the root)
- Best case, it combines the benefits of both DFS and BFS. Memory efficient and finds the solution closest to the root
- Worst case it's actually horrible
- **DP** is essentially backtracking search + memoization
- **Uniform cost search**: when exploring branches, remember the cost of the full path to the current node (from the root). When deciding, which node to expand next, expand the one with the lowest cost
- This is supposedly same as Dijkstra, but we stop when found solution (while Dijkstra explores every transition)
- UCS can be proven to be correct, and it potentially explored fewer states
- You do have overhead due to need to maintain the priority queue
- The cost of taking an action cannot be negative

## A* Search

- When deciding, which node to expand next, expand the one with the "best potential"
- Use some heuristic to `h(s->sEnd)` to estimate the future cost (the cost of the remaining path from `s` to `sEnd`)
- The search is as good as the heuristic
- For A* Search to be correct, the heuristic is required to be *admissible*
- A heuristic is admissible if it never overestimates the cost to reach a goal
- Basically, what this means: it's OK to examine some "bad branches", but it's never OK to ignore the "good branch"
- People usually care about consistency property: `h(s -> sEnd) <= Cost(s -> sX) + h(sX -> sEnd)` (TODO: for some reason)
- Every consistent heuristic is admissible
- To come up with heuristic, we relax some constraints
- Example: You need to find an optimal path walking from Joanic to Placa Catalunya, using pedestrian sidewalks and respecting the traffic lights. Relaxed constraint: you can just fly to Placa Catalunya. The real cost is actually walking the street. The heuristic is the direct distance

## Games

- Game tree: each node is a decision point for a player
- It's very very similar to search, but instead of cost, we are concerned with **utility** (gain). All of the utility is at the end state
- **2 player zero sum game:** given state `s1`, possible actions `A(s)`, next state `Succ(s, a)`, and agent's utility for end state `U(s)`, find a sequence of actions `a1, a2, ... an` that leads to the end state with highest agent's utility
- The "game board" is fully visible
- Example, chess: utility if infinity if you win, -infinity if you lose, and 0 if draw
- **Policy** is an action that player takes in state `s`: `P(s) -> a`
- Policies can be deterministic or stochastic
- Knowing the policy for both players, you are able to calculate **value** for each action `V(s, p)`
- In the end state, the value is just utility; it propagates up from the utility in a direction of the root, as an expected value from all the possible branches
- If you know the policy of the opponent, then your goal is, at every step, to find an action that maximizes the value
- **Minimax**: assume the opponent always picks the action that minimizes player's value. Pick an action that maximizes the value
- If opponent does follow the minimax, you cannot do better than minimax. But if the opponent actually does not follow minimax, you can do much better by not doing minimax
- Note that under minimax, the optimal path have the same value at each node
- Computational aspect is the hardest here: the branching factor and the tree depth often make it computationally impossible

### Addressing the computational aspect

- Option: limit tree search depth. When reached the max depth, instead of utility `U(s)` use some function `Eval(s)` that provides an estimate for the value of that state under minimax policy `V(s, minimax)`
- Maybe an expert could come up with some quick and easy rules
- This is similar how we used a heuristic in A* Search to estimate the future cost
- We can actually learn this function (e.g. NN)
- **Alpha-beta pruning**: get rid of (prune) the part of the tree, if you realize "there is no potential" going that path
- As you go down the tree, you maintain the upper and lower bound, and drop branches that don't overlap
- At each step my worst case is alpha and my best case is beta. If best case going right is worse than my worse case, never go right
- Example: there is a game, if you play well, then no matter what the opponents does, you are bound to win between 5$ and 1000000$. You seen 2 doors. The left door says: if you enter, you die. I think we all agree, there is no point ever going left
- On the other hand, if the door says "you either die or win 5.50$", you cannot exclude this door yet

## Constraint satisfaction

- Games care about order of actions, some problems don't
- Some problems only care about constraints (e.g. map coloring, event scheduling)
- You can still frame these problems as search problems, but the you would be ignoring some extra structure that comes with the problem
- __My thought__: extra constraints make problem more difficult to model, but allow for more efficient solution
- TODO: factor graph
- TODO: ways to solve factor graph