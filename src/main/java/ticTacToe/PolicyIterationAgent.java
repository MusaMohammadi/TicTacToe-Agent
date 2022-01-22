package ticTacToe;


import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		// ***** THIS CREATES A RANDOM POLICY ***** //
		//Create a entry set to retrieve the current set
		Set<Entry<Game, Double>> entryS = policyValues.entrySet();
		//Initialise a random operator
		Random rP = new Random();
		
		// loop over all the sets
		for(Entry<Game, Double> gs: entryS){		
			
			// Get the key corresponding to current game
			Game g = gs.getKey();
			// Store all the possible actions based on the current game in a list
			List<Move> actions = g.getPossibleMoves();
			
			// Loop until there are no more actions the agent can take
			if(actions.size() != 0) {
				
				// Create a list to store pairs
				List<IndexPair> pairs=new ArrayList<IndexPair>();
				
				// Modified from randomPolicy class
				for(int i=0;i<3;i++)
					for(int j=0;j<3;j++)
					{
						if (g.getBoard()[i][j]==' ')
							pairs.add(new IndexPair(i,j));
					}
				
				// Ensure the list contains pairs
				if(pairs.size() > 0){		
					// retrieve a random pair
					IndexPair random = pairs.get(rP.nextInt(pairs.size()));
					// Retrieve the random move associated with the random pair
					Move rMove = new Move(g.whoseTurn, random.x, random.y);
					
					// ensure the action is legal
					while (!g.isLegal(rMove)){					
						random=pairs.get(rP.nextInt(pairs.size()));
						rMove = new Move(g.whoseTurn, random.x, random.y);
					}
					
					// Add the game and random move to the current policy
					curPolicy.put(g, rMove);
				}
			}
		}
	}

	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the current policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)

	{
		// ***** THIS METHOD PERFORMS POLICY EVALUATION STEPS UNTIL THE VALUES UNDER THE CURRENT POLICY CONVERGE ***** //
		double maxDif;
		// loop until the maximum change is less than or equal to delta
		do{
			//Create a entry set to retrieve the current set
			Set<Entry<Game, Move>> entrys = curPolicy.entrySet();	
			// Initialise the maxDif
			maxDif = 0.0;
	
			// loop over all the sets
			for(Entry<Game, Move> g: entrys){			
				
				// Get the key corresponding to current game
				Game gs = g.getKey();
				// Store all the possible actions based on the current game in a list
				List<Move> actions = gs.getPossibleMoves();	
				// Retrieve the value of policy of the current game.
				double currentValue = policyValues.get(gs);			

				// If the current game is terminal, assign the policy value to 0.0
				if(gs.isTerminal()){
					this.policyValues.put(gs, 0.0); 
					}
				
				else{
					// loop over all the actions in the list actions
					for(Move b: actions){								
						// Retrieve the transition probability of the current mdp and store it in a list
						List<TransitionProb> tProbability = mdp.generateTransitions(gs, curPolicy.get(gs));	
						double vks1 = 0.0;		
					
						// Loop over the list of transition probabilities
						for (TransitionProb tp: tProbability) {
							
							// Retrieve the values to compute vks+1 = Sum of T(s,pi(s),s') [R(s,pi(s),s') + gamma * vk(s') 
							int ta = tProbability.indexOf(tp);
							TransitionProb currentTProbability = tp;
							double r = tProbability.get(ta).outcome.localReward;
							double t = tProbability.get(ta).prob;																		
							double sPrime = policyValues.get(currentTProbability.outcome.sPrime);
							vks1 = vks1 + t * (r + (discount * sPrime));															
						}
						// Calculate the difference between the current value and the new value 
						double dif = Math.abs(currentValue - vks1);
						// replace the old policy with the new one 
						policyValues.replace(gs, vks1);
						// Update the max Difference 
						if(dif > maxDif){
							maxDif = dif;
						}
					}
				}
			}
		// loop until the max difference is greater than or equal to delta 
		}while(maxDif >= delta);
	}
	
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		
		double max;
		Move bMove;
		Move currentMove;
		
		{
			boolean oPolicy = false;
			Set<Entry<Game, Move>> entrys = curPolicy.entrySet();	

			for(Entry<Game, Move> g: entrys){	
					// Initialise variables 
					max = 0.0;
					bMove = null;
					// Get the key corresponding to current game
					Game gs = g.getKey();
					// Store all the possible actions based on the current game in a list
					List<Move> actions = gs.getPossibleMoves();	
					// Create a list to store all the updated policy values
					List<Double> a = new ArrayList<Double>();	
					// Get the current move
					currentMove = curPolicy.get(gs);
					
					// loop over all the actions in the list actions
					for(Move b: actions){								
						// Retrieve the transition probability of the current mdp and store it in a list
						List<TransitionProb> tProbability = mdp.generateTransitions(gs, b);			
						double vks1 = 0.0;		
						
						// Loop over the list of transition probabilities
						for (TransitionProb tp: tProbability) {
							
							// Retrieve the values to compute vks+1 = Sum of T(s,pi(s),s') [R(s,pi(s),s') + gamma * vk(s') 
							int ta = tProbability.indexOf(tp);
							TransitionProb currentTProbability = tp;
							double r = tProbability.get(ta).outcome.localReward;
							double t = tProbability.get(ta).prob;																		
							double sPrime = policyValues.get(currentTProbability.outcome.sPrime);
							vks1 = vks1 + t * (r + (discount * sPrime));															
						}
						
						// add the newly calculated policy value to the list a
						// max stores the max of a
						a.add(vks1);											
						max = Collections.max(a);								
						
						// If the new value is the highest, make the current move the best move
						if(vks1 == max){										
							bMove = b;							
						}
					}
					
					// add the best move to the current policy and signify a optimal policy has been found, given that the current move is the best move
					if (!(bMove.equals(currentMove)))  {
						curPolicy.put(gs, bMove);
						oPolicy = true;
					}
				}
			return oPolicy;
		}

	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		{
			do {
				this.evaluatePolicy(delta);
			} while (this.improvePolicy() == true);

			super.policy = new Policy(curPolicy);
		}

		
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
