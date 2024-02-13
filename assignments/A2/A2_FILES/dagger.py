import bc
import torch

def interact(env, learner, expert, observations, actions, checkpoint_path, seed, num_epochs=100):
  """Interact with the environment and update the learner policy using DAgger.
   
    This function interacts with the given Gym environment and aggregates to
    the BC dataset by querying the expert.
    
    Parameters:
        env (Env)
            The gym environment (in this case, the Hopper gym environment)
        learner (Learner)
            A Learner object (policy)
        expert (ExpertActor)
            An ExpertActor object (expert policy)
        observations (list of numpy.ndarray)
            An initially empty list of numpy arrays 
        actions (list of numpy.ndarray)
            An initially empty list of numpy arrays 
        checkpoint_path (str)
            The path to save the best performing model checkpoint
        seed (int)
            The seed to use for the environment
        num_epochs (int)
            Number of epochs to run the train function for
    """
  # Interact with the environment and aggregate your BC Dataset by querying the expert
  NUM_INTERACTIONS = 100
  for episode in range(NUM_INTERACTIONS):
      total_learner_reward = 0
      done = False
      obs = env.reset(seed=seed)
      while not done:
        # TODO: Implement Hopper environment interaction and dataset aggregation here

        if done:
          break
      print(f"After interaction {episode}, reward = {total_learner_reward}")
      bc.train(learner, observations, actions, checkpoint_path, num_epochs)