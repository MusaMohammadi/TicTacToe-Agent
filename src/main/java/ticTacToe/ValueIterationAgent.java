package ticTacToe;


import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=50;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate()
	
	// ***** THS METHOD PERFORS K ITERATIONS STEPS TO DETERMINE THE VALUES OF EACH REACHABLE STEP ***** //
	{
		for(int j = 0; j < k; j++){
			
			/* Retrieve the current entry states */
			Set<Entry<Game, Double>> entryS = valueFunction.entrySet();

			/* loop over all the states */
			for(Entry<Game, Double> g: entryS){								
				
				/* Initialise a variable to hold the expectimax value */ 
				double max = 0.0;	
				
				// Get the key corresponding to current game
				Game gs = g.getKey();
				// Store all the possible actions based on the current game in a list
				List<Move> actions = gs.getPossibleMoves();	
				// Create a list to store all the updated policy values
				List<Double> a = new ArrayList<Double>();	
				// Get the current move
	
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
						double sPrime = valueFunction.get(currentTProbability.outcome.sPrime);
						vks1 = vks1 + t * (r + (discount * sPrime));															
					}
					
					// add the newly calculated policy value to the list a
					// max stores the max of a
					a.add(vks1);											
					max = Collections.max(a);									 
				}
				//clear list a
				// add the max value to the value function along with the game key
				a.clear();													
				valueFunction.put(gs, max);								
			}
		}
	}
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		// Create a new policy
		Policy p = new Policy();
		/* Retrieve the current entry states */
		Set<Entry<Game, Double>> entryS = valueFunction.entrySet();

		/* loop over all the states */
		for(Entry<Game, Double> g: entryS){								
			
			/* Initialise a variable to hold the expectimax value */ 
			double max = 0.0;	
			
			// Get the key corresponding to current game
			Game gs = g.getKey();
			// Store all the possible actions based on the current game in a list
			List<Move> actions = gs.getPossibleMoves();	
			// Create a list to store all the updated policy values
			List<Double> a = new ArrayList<Double>();	
			// Get the current move

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
					double sPrime = valueFunction.get(currentTProbability.outcome.sPrime);
					vks1 = vks1 + t * (r + (discount * sPrime));															
				}
				
				// add the newly calculated policy value to the list a
				// max stores the max of a
				a.add(vks1);											
				max = Collections.max(a);		
				
				// If the new value is the max, add the key of the current game state and the move to the policy
				if(vks1 == max) {
					p.policy.put(gs,b);
				}
			}
			//clear list a
			// add the max value to the value function along with the game key
			a.clear();																			
		}
		// return the policy
		return p;
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
