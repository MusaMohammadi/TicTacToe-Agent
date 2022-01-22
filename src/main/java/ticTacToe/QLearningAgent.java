package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.Map.Entry;
import java.util.Random;
/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to.
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=100;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair, you can do
	 * qTable.get(game).get(move) which return the Q(game,move) value stored. Be careful with 
	 * cases where there is currently no value. You can use the containsKey method to check if the mapping is there.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 30000, 0.9);

	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train()
	{
		// loop over all the episodes to train the agent
		for(int k=0; k < numEpisodes; k++) {
			
			// create a new environment for TTT
			TTTEnvironment env2 = new TTTEnvironment();
			Game g = env2.getCurrentGameState();
			Double updatedQValue = 0.0;
	
			// loop over the current game until it ends
			while (!g.isTerminal()) {

				// create a random double to compare to epsilon. This is used to determine whether to exploit or explore
				Random x = new Random();
				Double r = x.nextDouble();
				
				// if the random number is greater than epsilon, exploit
				if(r > epsilon){
						// ***** EXPLOIT *****//
					
						int maxAction = 0;
						// Initialise list of possible moves. If its empty, assign 0 to the QValue.
						// Else, assign a variable with the first instance of the list of actions.
						// This is used to compare the rest of the actions to find the max action.
						List <Move> actions = env2.getPossibleMoves();
						if (actions.isEmpty()) {
							updatedQValue = 0.0; 
							} 
						else {
							HashMap <Move, Double> valuesA1 =qTable.get(g);
							Move maxA = actions.get(0);
							updatedQValue = valuesA1.get(maxA);
						}
						// gets the maxA Q(s,a)
						// loops over all the actions, and retrieves the index of the action and QValue associated with the largest QValue from qTable
						for(Move m: actions) {
							if(qTable.getQValue(g, m) >= updatedQValue) {
								maxAction = actions.indexOf(m);
								updatedQValue = qTable.getQValue(g, m);
							}
						}
					
						// Execute the action with the largest QValue associated with it.
						Outcome outcome = null;
						try {
							outcome = env2.executeMove(actions.get(maxAction));
						} catch (IllegalMoveException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						
						// Create a list of actions based on the environment AFTER executing the action.
						// Initialise list of possible moves. If its empty, assign 0 to the QValue.
						// Else, assign a variable with the first instance of the list of actions.
						// This is used to compare the rest of the actions to find the max action.
						List <Move> actionsUpdated = env2.getPossibleMoves();
						if (actionsUpdated.size() == 0) {
							updatedQValue = 0.0;
						} 
						else {
							HashMap <Move, Double> valuesA2 =qTable.get(outcome.sPrime);
							Move maxA2 = actionsUpdated.get(0);
							updatedQValue = valuesA2.get(maxA2);
						}
						
						
						// Find the max of Q(s',a')
						// If s' is not terminal, complete the maxQ(s',a') loop, else assign the new QValue 0.
						// Loop over all the actions in the new actions list and retrieve the max QValue associated with S prime
						if(!outcome.sPrime.isTerminal()){
							for(Move nextA: actionsUpdated){
								if(qTable.getQValue(outcome.sPrime, nextA) >= updatedQValue) 
								{
									updatedQValue = qTable.getQValue(outcome.sPrime, nextA);
								}
							}
						}
						else {
							updatedQValue = 0.0;
						}
						
						// Calculate the sample using the formula 
						// Sample = R(s,a,s') + gamma * maxa'Q(s',a')
						// Where R signifies the Local reward, gamma is the discount and maxa'Q(s',a') is the maxQ value based on the max action taken.
						Double sample = outcome.localReward + discount * updatedQValue;
						// Add the new QValue onto the running average using the formula
						// Q(s,a) = (1 - alpha) * Q(s,a) + alpha * Sample
						Double QValue = (1 - alpha) * qTable.getQValue(outcome.s, (actions.get(maxAction))) + alpha * sample;
						qTable.addQValue(outcome.s, actions.get(maxAction), QValue);
				}
				else {
						// ***** EXPLORE ***** //
					
						// Initialise list of possible moves.
						List <Move> actions2 = env2.getPossibleMoves();
						// Create a random variable from 0 to size of the list of moves and retrieve the action associated with that number from the moves list.
						int upperbound = actions2.size();
						int l = x.nextInt(upperbound);
						Move a2 = actions2.get(l);
						

						// Execute the random action associated with it.
						Outcome outcome2 = null;
						try {
							 outcome2 = env2.executeMove(a2);
						} catch (IllegalMoveException e) {
							e.printStackTrace();
						}
						
						// Create a list of actions based on the environment AFTER executing the action.
						// Initialise list of possible moves. If its empty, assign 0 to the QValue.
						// Else, assign a variable with the first instance of the list of actions.
						// This is used to compare the rest of the actions to find the max action.
						List <Move> actions2Updated = env2.getPossibleMoves();
						if (actions2Updated.size() == 0) {
							updatedQValue = 0.0;
						}
						else {
							HashMap <Move, Double> moveValues =qTable.get(outcome2.sPrime);
							Move maxMove = actions2Updated.get(0);
							updatedQValue = moveValues.get(maxMove);
						}
				
						// Find the max of Q(s',a')
						// If s' is not terminal, complete the maxQ(s',a') loop, else assign the new QValue 0.
						// Loop over all the actions in the new actions list and retrieve the max QValue associated with S prime
						if(!outcome2.sPrime.isTerminal()){
						for(Move b: actions2Updated) {
							if(qTable.getQValue(outcome2.sPrime, b) >= updatedQValue) {
								updatedQValue = qTable.getQValue(outcome2.sPrime, b);
								}
							}
						}
						else {
							updatedQValue = 0.0;
						}
						
						// Calculate the sample using the formula 
						// Sample = R(s,a,s') + gamma * maxa'Q(s',a')
						// Where R signifies the Local reward, gamma is the discount and maxa'Q(s',a') is the maxQ value based on the random action taken.
						Double sample = outcome2.localReward + (discount * updatedQValue);
						// Add the new QValue onto the running average using the formula
						// Q(s,a) = (1 - alpha) * Q(s,a) + alpha * Sample
						Double QValue = (1 - alpha) * qTable.getQValue(outcome2.s, a2) + (alpha * sample);
						qTable.addQValue(outcome2.s, a2, QValue);
						}
				}
			
		}		
		
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null) 
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	
	}
		
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		// Create a policy to output
		Policy maxentry = new Policy();
		// Create a set to hold all the games
		Set<Entry<Game, HashMap<Move, Double>>> games = qTable.entrySet();
			
		// looping over games 
		for(Entry<Game, HashMap<Move, Double>> g: games) {
			
			// Initialise variables
			Double maxValue = -99.0;
			Move maxMove = null;
			// Loop over actions
			Set<Entry<Move, Double>> actions = g.getValue().entrySet();
			// Find the max QValue and extract the action and move associated with it to detemine an optimal policy
			for (Entry<Move, Double> a: actions) {
					if (a.getValue() >= maxValue) {
						maxValue  = a.getValue();
						maxMove = a.getKey();
						}
					}
			// Update the policy to hold the max values
			maxentry.policy.put(g.getKey(), maxMove);
			}
		// return the optimal policy
		return maxentry;
		
	}
		
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();

		
		

		
		
	}
	
	
	


	
}
