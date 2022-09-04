import os
#Transportation MDP. 
class TransportationMDP(object):
    def __init__(self, N):
        #N = number of blocks
        self.N = N
    
    def startState(self):
        return 1
    
    def isEnd(self, state):
        return state == self.N
    
    def actions(self, state):
        #returns list of valid actions
        result = []
        if state + 1 <= self.N: #If this condition is ture then walk is an option
            result.append("walk")
        if state*2 <= self.N: #If this condition is true then is taking the tram a valid option.
            result.append("tram")
        return result

    def succProbReward(self, state, action):
        #return list of new (newState, prob, reward) triples
        #state = s, action = a, newState = s'
        #prob = T(s, a, s'), reward = Reward(s,a, s')
        result = []
        if action == "walk": 
            result.append((state+1, 1., -1.))
        elif action == "tram":
            result.append((state*2, 0.5, -2.))
            result.append((state, 0.5, -2.))
        return result
    
    def discount(self):
        return 1.
    
    def states(self):
        return range(1,self.N+1)

mdp = TransportationMDP(10)
print(mdp.actions(3))

#Inference (algorithm) to get best optimal policy
def valueIteration(mdp):
    #MDP: is our markow decison process model/statement
    #Epsilon: The degree og error we allow between V[state,t-1] and V[state, t] before we accept convergence.
    V = {}
    for state in mdp.states():
        V[state] = 0 #state -> V_opt[state]
    
    def Q(state, action):
        return sum(prob*reward + mdp.discount*V[newState] \
            for newState, prob, reward in mdp.succProbReward(state ,action))

    while True:
        # compute the new values given the old values (V)
        newV = {}
        for state in mdp.states():
            if mdp.isEnd(state):
                newV[state] = 0.
            else:
                newV[state] = max(Q(state, action) for action in mdp.actions(state))
        # check for convergence
        if max(abs(V[state]-newV[state]) for state in mdp.states())<1e-10:
            break
        V = newV 

        #Read out policy
        pi = {}
        for state in mdp.states():
            if mdp.isEnd[state]:
                pi[state] = "none"
            else:
                pi[state] = max((Q(state, action),action)  for action in mdp.actions(state))[1] #itterate over all actions given our current state. Which action then maximize Q.
        #print stuff out
        os.system('clear')

        print('{:15} {:15} {:15}'.format('s', 'V(s)', 'pi(s)'))
        for state in mdp.states():
            print('{:15} {:15} {:15}'.format(state, V[state], pi[state]))
        input()

valueIteration(mdp)