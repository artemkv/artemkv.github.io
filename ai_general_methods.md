# AI General Methods

## References

- [MIT 6.034 Artificial Intelligence](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi)

## Intro

- What is intelligence?
- If computer can imitate something that human does when solving a particular problem (e.g. symbolic integration), is that intelligence?
- When you understand how exactly the algorithm works, your perception of its intelligence tends to go to zero
- Does series of if-else operators capture what we call "intelligence"? Does it matter?
- Maybe there are different types of intelligence
- Learning philosophy: to build the skill, you need to understand one level deeper; to understand you need to witness one level deeper
- One very simple idea for the problem solving approach: **generate and test**. Basically, you generate multiple potential solutions and test whether they are correct


## Goal Trees

- **Problem reduction**: apply transformation to a harder problem to make it an easier problem
- Symbolic integration algorithm tries to apply 1 of many possible safe transformations (e.g. move constant out) and heuristic transformations (e.g. substitute `x` with `sin(u)` in expression `1-x^2`), then checks against the known logarithms table, if success, reports the solution; if not, continue
- Sometimes you may be able to apply several transformations, so the tree of possibilities begin to form
- To decide which branch to pursue, you may decide to go for the simplest one first
- How do you measure simplicity?
- In this case you could compute the depth of functional composition
- This program was running on 32K of RAM and was able to solve 52 out of 54 problems that it was given (the problems were taken from freshman test for MIT students)
- Max depth of the tree was 7, average depth was 3, and average number of unused branches was 1
- This gives quite a good insight into the nature of the domain (freshman level integration problems)
- This kind of meta knowledge is where the power is

### Answering questions about its own behavior

- Imagine the program that can move the blocks of various sizes laying on the table
- You may ask it to put block `X` on top of a block `Y` (e.g. "`PutOn(b1, b2)`")
- Block `X` may have some other blocks on top, so you would first need to get rid of those
- Block `Y` may have some other blocks on top too
- So, the top-level sequence of `PutOn` is: `FindSpace, Grasp, Move, Ungrasp`
- `FindSpace`: call `GetRidOf` in a loop
- `Grasp`: `Cleartop` by calling `GetRidOf` in a loop
- The complexity is in the `GetRidOf`, which has to call into `PutOn` recursively
- If you trace the execution of this program, you will construct the goal tree, similar to what we had in the case of the symbolic integration
- Now, having that tree, you can ask questions
- To answer "why" questions, you need to go up the trace: "why did you want to get rid of b5?" → "because I wanted to clear the top of b1"
- To answer "how" questions, you need to go down the trace: "how did you clear the top of b1?" → "by getting rid of b5"
- The tree can grow pretty big and have lots of branches
- However, this is not due to the complexity of a program (which is pretty simple), but the complexity of the problem

```
The complexity of the behavior is the Max of (complexity of the environment, complexity of a program)
```


## Rule-Based Expert Systems

- Example: system that recognizes an animal by its characteristics
- "Has hair?" → "It's a mammal"
- "Hash claws?" & "Has sharp teeth?" & "has forward-pointing eyes?" → "It's a carnivore"
- "Is it a mammal?" & "is it a carnivore?" & "has spots?" & "is it very fast?" → "It's a cheetah"
- This system would be a forward-chaining, as it goes from the left to the right
- This is also a goal tree, so it can also answer questions about its own behavior: "how did you know that the animal is a mammal?" → "because it has hair"
- And in the opposite direction, "why did you ask whether the animal had hair" → "because I wanted to determine whether it was a mammal"
- You can also run the system backwards: backward-chaining
- How to build those systems?
- If you just go and talk to the experts, all you'll get is generalities, because they will not think of everything
- So you need to go and watch the experts in action
- Consider the task of bagging the groceries
- If you talk to experts, they might say something like "heavy items go to the bottom", but hardly anything else
- Rule #1: look at specific cases!!! This will make people give you knowledge they otherwise would not think of
- E.g. see how they actually pack specific items, e.g. bag of chips, pack of pasta
- Rule #2: ask questions about things that appear to be the same, but are handled differently
- E.g. ask why they handle canned peas differently from frozen ones
- Rule #3: build a system and see where it cracks, this will help you to identify a missing rule
- E.g. you go and try to pack all possible items you may find in the supermarket

```
How to speak to domain experts:
- Don't speak in general terms, look at them dealing with specific cases
- Ask questions about things that appear to be the same, but are handled differently
- Build a system and test to identify missing rules
```


## Search (DFS, Hill Climbing, Beam, Branch&Bound, A-star)

- The discussion is about the search on the graph, e.g. finding a path on the map (weighted or unweighted)
- **British museum** search: look at every possible path between the start and the end node (brute force)
- This is a base case against which all the other search is compared
- **Depth-first** search: when you build the tree of choices, you always explore the leftmost branch first (until you see the end node, or until you reach the dead end and backtrack)
- **Breath-first** search: you build the tree of choices level by level, until you see the end node
- DFS and BFS is almost the same implementation, the only difference is where you put the expanded nodes on the queue (front or back)
- The great optimization is not to expand any node twice (by keeping the nodes we already visited in a separate list)
- **Hill climbing**: when you expand the node, always go into the one that gets you closer to the goal, discard all the rest
- You just need to use some kind of heuristic that can tell you whether you are getting closer to the goal (so-called informed search)
- In case of searching for the path on the map, you could measure direct distance between 2 points (how a crow would fly from A to B)
- In general, this would result in being useful, even though, in some cases, this would lead you to the dead end
- This is very similar to gradient ascent, but in hill climbing you don't deal with gradients, you just try different options and see which works the best
- Problems are also very similar: you can get stuck in a local maximum or in a flat area; sometimes you can get fooled thinking you are on the top while you are at the ridge
- **Beam** search: similar to hill climbing, you limit the number of nodes you expand, picking the most promising ones, using a heuristic, but instead of expanding just one, you may expand 2 (beam of size 2) or more
- **Branch and bound**: always expand the node that has the shortest accumulated length, until you find the end node, remember that length. Then keep expanding all the paths that are still shorter than the one you found, until you find a shorter one or go beyond the shortest length currently found
- Implementation is still similar, just keep the queue sorted by accumulated length (or use a priority queue, or simply find a node with the shortest path every time you need to pick one from the queue, which is `O(n)` while sort is `O(n*log(n)))`
- Powerful optimization: don't extend what's already been extended (meaning, queue it still, but discard if already extended)
- This algorithm is also great if you found some candidate path, using some kind of heuristic, and just want to check whether there is a shorter path. You would bound by the length of the candidate path
- Notice that, at any point, the airline distance to the end node gives you a lower bound for the shortest path (can't get there quicker than by going straight to it)
- So when you expand, for any candidate subpath, you can evaluate a lower bound of the full path as a length of that subpath + airline distance to the end node, and you can expand in that order
- Airline distance would be admissible heuristic in this case. To be admissible, it has to be guaranteed to be ≤ than actual distance
- **Admissible heuristic** `H`: `H(x→G) ≤ Distance(x→G)`
- **A-star** search is putting the 2 ideas together: don't extend what's already been extended and use an admissible heuristic. Boom! This outperforms both optimizations applied separately
- But beware!! There exist corner cases where admissible heuristic in combination with only extending every node once actually leads to finding a suboptimal path
- To make it work 100%, we need a new restriction for the heuristic, namely, it has to be consistent (in addition to be admissible)
- **Consistent heuristic** `H`: `|H(x→G)-H(y→G)| ≤ Distance(x→y)`
- Cool use of search: look for a high-level pattern in a text
- For example, "revenge" pattern: X harming Y leading to Y harming X, look up for that in "Lady Macbeth"


## Search (Minimax, Alpha-Beta)

- How could you teach computer to play chess? One valid approach could be "look ahead and evaluate"
- Basically, at any point, build a complete list of legal moves and try to see which one looks "the most promising" (evaluate)
- Evaluating typically means looking at some list of features `f1, f2, ..., fm` and building a polynomial `c1*f1 + c2*f2 + ... + c3*f3` (a score)
- With branching factor `b` and `d` steps lookahead, you would need to evaluate `b^d` positions
- You could try to look 50 steps ahead
- For a game of chess, `b`, on average, is ~14, and if you are looking 50 moves ahead for each player, d is 100, so every time you need to evaluate 14^100 positions, so it's something like 10^120 evaluations
- Turns out, there is ~10^80 atoms in the universe
- The age of the universe in nanoseconds is in order 10^26
- So if every atom in a universe was doing the computation at the nanosecond speed, starting at the big bang, we would still not be able to finish evaluating even a single move
- Realistically, in chess, you would need to go 15-16 levels deep to beat the world champion, and if you go that deep, the only feature that matters would be the piececount. But that is still pretty costly
- So all we can do is to branch as deep as possible
- This is the **Minimax** game setup: 2 players, tree of possible moves, each leaf gets evaluated, the evaluation is propagated back to all upper nodes, 1 player is trying to maximize that value, another one to minimize
- When you do that, you need to keep track, for each of tree depth levels, whose move it is, maximizer's or minimizer's, and propagate either min or max
- **Alpha-Beta** is the layer of optimization on top of minimax
- The idea is to minimize work by skipping parts of the tree that don't lead to a better path, using upper and lower bounds
- Imagine you are maximizer, and you are analyzing your opponent's move several steps into the future. Let's say that move results in the score of 2. That might be the best you can do, sure, but now you know that, whatever move you make, your worst possible score is ≤2 (in that whole subtree)
- Now let's get 1 level up, now analyzing your move. Even before looking at all the options, you know, that, at worst, you can get at least 2, i.e. your best possible score is >=2
- This means that, as you continue analyzing, you can immediately discard all the branches that are guaranteed to give you the score <2 ("dead horse principle")
- Using alpha-beta, the claim is, you may be able to find an optimal move in just `2b^(d/2)` evaluations, comparing to `b^d` evaluations using minimax
- In practice, it depends on the data, but it seems that, on average, you would be pretty close to the optimal case
- Note on the algorithm: very similar to minimax, but you propagate lower/upper bounds in addition to an actual score. The propagation rules are very simple, but it's very difficult to wrap you mind around this algorithm and convince yourself it actually works (lower/upper bound statements are not intuitive)
- Typically, the branching factor depends on how far you are into the game, so you cannot actually predict how many levels deep you might be able to go in a limited amount of time
- Unfortunately, with minimax, you start from the leaves, so in many cases you would be either too fast (leaving a better move on the table) or too slow (not being able to finish the analysis and come up with a move in a specified time)
- But notice that amount of moves to analyze on each level increases by a factor of `d`; that means that doing a calculation up to level `d-1` is `d` times cheaper than doing full calculation; i.e. it would cost only `1/d` of the full cost
- This means you could do the calculation with `d-1` steps lookahead and use that as an "insurance policy" in case you run out of time doing the full calculation, that would cost you just a `1/d` fraction of the total cost; then try to do the full calculation for d steps lookahead, and if you can, use that better answer; if not, use the insurance policy answer
- Actually, you could start with just depth of 1, then, if you still have time, go 1 level deeper, and so on, as deep as you can, until you are out of time: **progressive deepening**
- But at what extra cost?
- Cost `S` of progressive deepening up to the level `d-1` is `1 + b + b^2 + ... + b^(d-1)`
- Massaging these expressions, we get:
- `b*S = b + b^2 + ... + b^d`
- `b*S-S = b^d-1`
- `S = (b-1)*S/(b-1) = (b*S-S)/(b-1) =` (using the math above) `= (b^d-1)/(b-1) ~= b^d/b = b^(d-1)`
- This means that, with total cost for `d` steps lookahead being `b^d`, the total amount of extra computation doing progressive deepening would still be around `1/d` of the full cost of calculation for just layer `d`
- Basically, going up the three, the amount of computation shrinks so fast, it almost doesn't matter (another quite unintuitive, but very beautiful concept)
- This produces so-called "anytime algorithm", the one that is always ready to give you an answer, regardless of how much time it is given
- Deep blue (beat Kasparov in 1997): minimax + alpha-beta + progressive deepening + parallelization + some other ideas
- Deep blue is a kind of "bulldozer intelligence"


## Constraint propagation: interpreting line drawings

- Given a contour lines of objects, detect how many distinct objects are there
- **Experimentalist approach** (Guzman): play with real objects, take a lot of photos, look at them, notice the different types of line junctions. Figure out there are 2 types of line junctions formed by surfaces belonging to a same object ("arrow" and "fork"). Count the amount of junctions of these types (links) between any 2 surfaces. Come up, empirically, with minimal number of links that are required to conclude the 2 surfaces belong to the same object
- **Mathematician approach** (Huffman): forget the real world, construct your own artificial world, but in precise math terms. Impose strict constraints (i.e. discard anything you don't like). Consider 4 types of lines and discover that there is only 18 ways to arrange a junction in this world, using those 4 types of lines. This catalog allows you to verify constructability of any object (if it's not in the catalog, you can't build it, in this world)
- None of 2 approaches actually solves a real world problem of computer vision
- Guzman was solving a right problem, but he didn't have a good method (too ad-hoc)
- Huffman had a method, but didn't solve an actual problem
- **Engineer's approach** (Waltz): use the math method, but apply it to the full complexity of the real world. Consider cracks, shadows, vertices constructed of more than 3 surfaces, light. Discover this leads to 52 types of lines, and thousands of types of junctions. Invent the algorithm to interpret any object without the need to expand the complete tree of possible junctions
- The intuition behind an: when you are trying to guess what's on the picture, you may look at all the individual parts, and think what kind of objects they could appear on (e.g. window can be a part of a car or a house, and a wheel can be a part of a car or a wheelchair). So you could try to expand every possibility, but that would be potentially millions of possibilities
- Instead, you propagate constraints every time you move to the next part (e.g. wheel can be a part of a car or a wheelchair, but not a house, so when you see a window, discard house already. Then propagate back, and discard a wheelchair, and voilà, there is only 1 possibility left)


## Constraint propagation: map coloring + domain reduction

- Imagine you are trying to color the US map using 4 colors: RGBY, you go over the list of states in some order and try to apply colors in a round-robin fashion
- If you discover a conflict, you try next color, and if you are out of colors, you back up one step
- Problem is, if you are unlucky, and a state like Texas comes somewhere in the end, you might get stuck forever
- Why? Imagine that, when you arrive to Texas, you have already colored New Mexico, Oklahoma, Arkansas and Lousiana using 4 different colors, so now there is no color left that you could use for Texas, and you have to back up
- But then, if you are unlucky with the order, the problem might repeat again and again, every time you arrive to Texas
- Moral of the story: undiscovered local constraints cause downstream problems
- To avoid this issue, every time you pick a color for any of the states, you need to consider states local to it, such as Texas, even though they are coming later on the list
- "Consider Texas" means every time you pick a color, you remove that color from the list of color Texas will be able to use once we get to it (its domain)
- Let's say you color New Mexico red, Oklahoma green and Arkansas blue; so now Texas only has yellow
- Once you get to Lousiana and try yellow, you'll see nothing is left for Texas; so instead you'll immediately try red
- Red will work for Arkansas and will leave yellow for Texas
- Generalizing this algorithm, when you pick a color for a current state, you need to consider all the neighbor states and adjust their domains by removing that color; and then check that neither any of them nor the state itself haven't run out of colors; and if it happens, back up before going any deeper (**Domain reduction algorithm**)
- You actually have a choice as of how many other states to consider
- Instead of considering only direct neighbors, you can propagate the check to the neighbors of neighbors, up to a certain depth
- The more you propagate, the more extra work, but also more chances to detect conflicts earlier
- It turns out to be very efficient to propagate only through neighbors that have their options (domains) shrunk to just one value
- Being "unlucky" with the order is the key. Turn out, if you arrange states from most constraints to least constraints, you don't need all this trickery, and normal DFS would work just fine
- So, in the real world, do both things: sort and do domain reduction
- Same problem, different context: scheduling problem
- E.g. you are given the time schedule of `M` flights, and `N` aircrafts, how to assign aircrafts to flights? The constraint is: one aircraft cannot serve 2 flights at the same time
- If you want to find out `Nmin`, the minimal number of aircrafts you need, you can quickly estimate the range, by trying low and high numbers. Lowest possible is 1, the highest possible is `M`. Doing some simulations on numbers in between, it will either complete really quickly or give up really quickly, so you will be able to narrow the approximate range quite fast


## Constraint propagation: visual object recognition

- Failed idea: identify the contour lines, do several "helpful" transformations, compare the result with the library of known objects
- Too hard, too imprecise
- Another idea: if you have orthographic projection of a 3d object on every axis (i.e. 3 projections), you can construct an orthographic projection of that object at any angle
- All the shape-preserving transformations (e.g. rotation) can be expressed in a way that would be exactly the same for all the points of an object, and actually be linear
- E.g. with rotation by some angle theta, all the points will displace `x*cos(theta)-y*sin(theta)`
- And if you have enough projections of the same objects, you can figure out constants `cos(theta)` and `sin(theta)` by solving the system of equations (take 3 projections, pick any 3 points and solve)
- Once you know the constants, you can verify whether the shape you are looking at is a different projection of the same object or is a different object, by verifying the predicted position of all points
- Yet another idea: compute a correlation between 2 images
- You would need to use intermediate size features to correlate
- E.g. the whole face would probably be too big to get a good correlation, individual features like eyes can produce good correlation, but may appear at incorrect position in relation to other features (nose, mouth), eyes + a nose seem to be good compromise
- This method works incredibly good finding a known face on a larger image, even when a lot of noise is added
- But, unfortunately, it has huge limitations too, and this is probably not how our visual apparatus works


## Genetic Algorithms

- Similar to neural networks, this approach is mimicking the biology, in this case, evolution
- Given function `f(x,y)`, find the optimal value of `x` and `y` that maximize the function
- Each "chromosome" will contain values for `x` and `y`, e.g. `[x=0.3; y=0.7]`
- Mutation will modify a value of either `x` or `y` component, e.g. `[x=0.3; y=0.7] → [x=0.2; y=0.7]`
- Crossover will exchange the components between 2 "chromosomes", e.g. `[x=0.2; y=0.7] and [x=0.6; y=0.2] → [x=0.2; y=0.2] and [x=0.6; y=0.7]`
- You need to estimate **fitness** of every individual, some positive real number
- Use fitness to calculate the **selection probability** (probability of survival)
- Define the size of the population and run the algorithm for a given number of iterations
- Notice that, if some way, this is very similar to hill climbing: you are trying various directions and pick the one that you consider to be the best, based on fitness
- There are many parameters you can tweak to make this simple algorithm perform better

### Selection methods

- In **Roulette Wheel Selection**, the selection probability is directly proportional to the individual fitness: `fitness_i/(sum of fitnesses for all i)`
- This method largely favors the current fittest individual, which may lead to premature convergence to a local optimum
- It also has some weird properties, as it depends on the scale you are using, and where you put the zero (consider +1 and +10 on the Celsius scale become +274 and +283 on the Kelvin scale, and if you are using the temperature as a measure of the fitness, it's easy to see how the change of scale would produce dramatic difference in calculated probabilities)
- In **Rank Selection**, the selection probability does not depend on the fitness directly. Instead, the fitness is used rank of an individual within the population. We consider every individual in that order, and give every individual the same probability `Pc` to reproduce
- The probability of the first ranked individual to reproduce is `Pc`
- The probability of the second ranked individual to reproduce is `(1-Pc)*Pc`
- The probability of the last ranked individual to reproduce is `(1-Pc)^(N-1)`, i.e. if no one else is selected, the last individual is guaranteed to be selected
- Rank-based selection preserves the diversity, it gives worse individuals a chance to reproduce and thus to improve
- On the flipside, it can lead to slower convergence, because there is not much difference between the best individuals
- Looking at the classroom example, turns out, it can still stick at local maximum
- _My thought: when you model biology, you should not expect results to be different from those that biology produces. In the best case, you are going to be as good as biology is. And I don't think biology cares about being stuck in local optimum_
- We could factor in the diversity factor: don't select just the most fit, but most fit AND most diverse
- This helps to discover the true optimum fast, however, it never really converges, as the diversity keeps is spread

### Applications

- Planning: instead of optimizing `x` and `y`, you may want to optimize a series of steps `[s1, s2, s3, ..., sn]`
- Each "chromosome" then represents a potential sequence
- In general, you can take any long string of parameters `[p1, p2, p3, ..., sn]` and optimize them using the genetic algorithm
- The algorithm performs the best when there is a rich space of solutions (e.g. whatever structure made of blocks may learn how swim, and there is going to be enormous variance in how different creatures do it, depending on what blocks they are made of)
- Back in 2010, both NNs and genetic algorithms were considered "low worth" by MIT


## Sparse Spaces

- Instead of trying to focus on the method (or a specific algorithm) to solve a problem, we can try to focus on a problem itself
- Basically, powerful and super-flexible generic tool (like NN) vs super-specialized tool for a specific problem
- Trying to apply NN to every problem is probably stupid, and instead you should start with the problem
- So the approach should be: define the problem → find a right representation → find a method → develop the algorithm → experiment
- Right representation is the one that "makes important things explicit", exposes constraints and satisfies **localness** criteria (all the important things that are related are close together)
- Example of a problem in phonology: "s" in dogs and "s" in cats are pronounced differently (first is "z" and second is "s"), how do you learn that?
- How do you even make sense of a sound?
- 2 people saying the exact same sentence would produce 2 acoustic waves that would look extremely different
- You can, however, distinguish about 14 binary features that would describe a sound, based on the use of "pieces of meat" that form your mouth (e.g. whether it is labial or dorsal, nasal or strident etc., see "Phonetic Features")
- This would allow, in theory, for more than 16K combinations; in practice, languages use much less (about 40 in English)
- Some features are "hallucinated": they are not actually found in a waveform, but appear because of the meaning we attribute to the phoneme (i.e. the meaning feeds back into the input for the feature detection)
- There is also injection from other modalities, e.g. what you see can impact what you hear (see "McGurk effect")
- The reason many people learning foreign languages have trouble speaking like a native is: they are not watching the mouth of a speaker
- The machinery: visual image of 2 apples, detect the word ("apple") and morphology (noun, plural), apply constraints ("plural and ends in voiced" → "z"), put in the output buffer to produce the stream of phonemes
- The information is allowed to flow in both ways between the boxes (that's why they are called "Propagators")
- The question is then: how do you build a set of constraints like this?
- The answer is: collect a lot of positive negative examples (e.g. "KATS", "DOGZ", "KATZ", "DOGS"), look at all 14 features for every of 4 phonemes (14*4 matrix), and try to figure out the rules (generalize)
- You want to find determining features that separate positive examples from negative examples (minimal subset of features that still retains enough information about how the 2 words are different)
- You'll discover that this is a hard task, as there are too many features
- The solution: pick one positive example to be a **seed** (e.g. "KATS")
- Then we start generalizing, by "turning off" 1 of the cells in the 14*4 matrix, until we cover (or "match") a negative example, and then we quit
- The order we pick cells to turn off is: from the ones that are the closest to the phoneme that we are trying to build the rule for, to the ones that are the furthest
- There are theories on why it works
- One explanation: this is a sparse high-dimensional space of features, and it's very easy to find a multidimensional hyperplane that separates positive and negative examples in such a space


## One-shot learning (by analyzing differences)

- This comes from Patrick Winston's early research (a professor teaching this class)
- Humans don't learn from millions of examples; quite the opposite, we often learn something from just one example
- Example: you show a martian a table and tell him: "this is a table". Then you remove a tabletop and tell him: "this is no longer a table". Then you put that back and remove one leg instead, and say: "this is still a table". And every time you do that, a martian would learn something new about the concept of a table, and have a better and better understanding of it (**evolving model**)
- We will do that by establishing different links between things you observe, and qualifying those as one of 6 types: require-link, forbid-link, climb-tree, enlarge-set, drop-link and close-interval
- **Initial description**, a starting point: "4 legs and a surface on the top, supported by those 4 legs"
- **Near miss**: a negative example that is very close to a class that we want to learn, but not quite: "4 legs and a surface in a pile"
- Comparing 2 examples, you'll immediately realize that the "support" relationship between the legs and the surface is important: **require-link**
- We continue with more near misses, gradually improving our model, e.g. "4 legs and a ball on the top, supported by those 4 legs" (**forbid-link**)
- We can also continue with examples that are very close, but don't change the nature of an object: "4 legs and a surface on the top, supported by those 4 legs, but this time the surface is round instead of rectangular" (**enlarge-set link**)
- Now you know the surface can be rectangular or round, and you may even generalize it to "surface of any shape" (**drop-link**), depending on your world (but you can also wait for 1-2 more example before generalizing)
- So we learn some new rule with every example, and this is the striking difference compared to NN, where we need thousands of examples to learn anything
- Ideally, you carefully pick every subsequent example to drive home another point; one well-chosen example can replace extended and tedious training process
- The teacher should provide negative examples and near misses along the positive examples, otherwise you would not be able to isolate what is important and learn
- Learning should be done in small steps
- You can use this technique to understand the difference between 2 types of objects that are quite complicated in their structure and look very similar to a non-expert (or even to an expert!)
- Example is trying to predict a function of protein based on their primary, secondary and tertiary structures
- If we are learning some system produced by humans, there are always going to be exceptions and special cases, those have to be handled in a special way
- _My thought: I think this idea is underrated in 2024, but... I think in the past, this is how I learned languages, and in 2024 I learn languages more like a NN. This stuff just doesn't work too well with languages like French, which are full of exceptions_

### Procedure

- a) Collect several positive and negative examples
- b) Pick one positive example as a **seed**
- The description of a positive example initially would consist of all the features of a seed. Because of that, no other example would match this description exactly
- c) Pick one feature (heuristic) that would relax this description to cover more positive examples
- This also may include some negative examples, but that's ok, as you may decrease the number of negative examples at the next step
- d) Keep picking one more feature, so that more positive and less negative examples are covered
- By doing so, you would be expanding the tree of features, and it would be an extremely wide tree. Beam search technique to the rescue
- At every step, you would be either specializing or generalizing
- _My note: I guess, you stop when you are able to separate positive examples from the negative ones; but maybe we just stop when we are done with all the examples and learned as much as we could from those_

```
How to package your ideas

- Symbol: use visual symbol to represent your idea
- Slogan: doesn't explain the idea, but puts you in the same mental state as if you understood the idea ("near miss")
- Surprise: different from most of the things done/expected in the field
- Salient: the idea that sticks out (don't pack too many good ideas together)
- Story: people love stories
```


## Boosting

- Any binary classifier can be benchmarked against a coin flip
- **Weak classifier** does "just slightly" better than a coin flip
- The key question: can you make a strong classifier by combining multiple weak ones, and letting them vote?
- It's easy to see that, if every weak classifier makes mistakes on a subset of training examples, but they don't overlap, a simple majority vote would completely eliminate all the mistakes
- That is an ideal scenario, in reality those subsets can overlap and even overlap in such a nasty way as to give worse performance than any individual weak classifiers
- But we can try and be smart about training these classifiers in order to minimize the overlap in subsets of incorrectly classified examples
- We can make a strong classifier `H` out of weak classifiers `h1, h2, ..., hn` that we are going to train and select one by one
- These classifiers can be of any type; for illustration purposes, people usually pick a single test of a decision tree as a classifier
- We will first train a bunch of classifiers and pick `h1` to be the one that performs best on the original dataset
- We will then train a bunch of classifiers and pick `h2` to be the one that performs best on the weighted dataset; extra weight being given to the examples on which `h1` failed
- We will then train a bunch of classifiers and pick `h3` to be the one that performs best on the re-weighted dataset; extra weight being given to the examples on which `h1` and `h2` produced different results
- So, we are basically driving the selected classifiers to have a desired property of not having error overlaps
- Now, if we take these 3 weak classifiers `h1`, `h2` and `h3`, and combine them, it will produce a new weak classifier `h1'`, hopefully, somewhat better than any of these 3 separately
- We can treat this new classifier `h1'` as a single unit and repeat all the same steps; this will produce a tree of classifiers with `H` on top, branching in triplets
- This was state of the art for many years, until Freund and Schapire discovered Adaboost
- **Adaboost** is a specific variation of boosting that allows to build a strong classifier out of an unbounded sequence of weak classifiers, determined iteratively, one by one; and the final classifier `H` will correctly classify all samples in the dataset
- Just as before, we will first train a bunch of classifiers and pick `h1` to be the one that performs best on the original dataset
- Then on every step `i`, we will train a bunch of classifiers and pick `h_i` to be the one that performs best on the re-weighted dataset; extra weight being given to the examples on which `h_i`-1 failed; and so on
- We stop once the classifier `H` correctly classifies all samples, or you cannot find any weak classifier for the next step
- The specifics of Adaboost are in the way it calculates weights and multipliers for `h1, h2, ..., hn`, and how it combines them all together in the final formula
- Adaboost is based on a very formal math and can be analyzed; specifically, it can be proven that the lower bound of the classification errors declines exponentially
- For some reason that is not perfectly understood, this method seem not to overfit while being able to handle complex decision boundaries


## Representations

- We can construct a **semantic net** from the text
- In such a net, Macbeth is connected to Duncan with an arrow tagged as "murder"
- "Murder" requires agent (Macbeth), object (Duncan) and an instrument (knife)
- All this stuff seems to be kind of stuff linguists like Chomsky try to think about
- E.g. "a lot of our thinking involves an object moving along a trajectory, from a source to a destination"
- Then you can link more descriptions to those elements
- "From" tag to a source, "to" tag to the destination, "by" to the agent, "with" to co-agent, "for" to a beneficiary etc.
- Once you have done that, you can notice some patterns, search for analogies etc.
- High-level patterns form stories, and you can have libraries of stories


## Bayesian networks

- Imagine you have 5 variables: "burglar", "racoon", "dog barks", "trashcan open", "call the police"
- You may record frequencies of events made of every combination of each of these variables in one huge table of joint probabilities (2^5 rows for 5 variables)
- This table can be super-useful to extract new insights (such as conditional probabilities, making bayesian inference etc.)
- However, working with such a huge table is unwieldy
- **Belief net (Bayesian network)** is a directed acyclic graph, representing conditional dependencies between events
- Belief nets capture causality between events
- Example: "Burglar" → "Dog barking", "Racoon" → "Dog barking"
- The constraint: every event is conditionally independent of all non-descendents (conditionally: given its parents)
- _TODO: it's not clear why and how it could depend on descendants_
- In other words, all the causality is flowing through the parents
- Because of that, if we assign probabilities to events in such a net, we would need to keep track of significantly fewer numbers (every event would have a smaller table of probabilities associated with it, having only parents as variables)
- All of this is possible because, as humans, we can make statements on what we believe depends on what (and this is how we build this kind of net)
- It can happen though, that different people come up with different nets, because we have different beliefs
- But we can calculate the probability of each model given the data (i.e. likelihood), and see which one is more likely (exactly the same way we estimate how likely our coin is fair based on series of tosses): **model selection**
- We can use this to actually build net from scratch, without need to put in any prior belief
- Brute-force approach would be to just try all possible nets and see which one fits data the best, but this would be too inefficient
- Instead we can use (random) search
- We start with 2 initial models, the winner stays, the loser is replaced with the winner modified slightly (and randomly)
- To avoid being stuck in a local maximum, you do some crazy re-arrangement of a loser with a small probability
