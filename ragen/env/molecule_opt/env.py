import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict, Optional
import math # Add math import for ceil

# --- Cheminformatics Imports ---
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not found. Molecule modification and similarity calculation will not work.")
# -----------------------------

from ragen.env.base import BaseLanguageBasedEnv # Using LanguageBasedEnv as modifications might be textual
from .config import MoleculeEnvConfig
from .oracle import MoleculeOracle # Import the oracle
from .property_utils import (
    parse_task_config, 
    validate_task, 
    calculate_multi_objective_reward, 
    generate_complete_multi_objective_guidance,
    get_optimization_directions  # Get optimization direction for each property
)


class MoleculeEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config: Optional[MoleculeEnvConfig] = None, env_config: Optional[dict] = None):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for MoleculeEnv. Please install it (`pip install rdkit`).")

        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else MoleculeEnvConfig()
        
        # Handle task configuration and variable substitution
        self.env_config = env_config or {}
        self._parse_task_config()
        
        # Set target_property based on molecule_opt_task
        task_properties = self.molecule_opt_task.split('+')
        if len(task_properties) == 1:
            self.target_property = task_properties[0]
        else:
            self.target_property = task_properties[0]
            
        self.optimization_directions = get_optimization_directions(self.molecule_opt_task)
        self.minimize_target = self.optimization_directions.get(self.target_property, "maximize") == "minimize"

        # --- Validate dynamic threshold config ---
        if not hasattr(self.config, 'initial_similarity_threshold') or \
           not hasattr(self.config, 'final_similarity_threshold'):
             # max_steps (episode steps) is still needed for termination check, but not for threshold calc
             # total_training_steps will come from options during reset
            raise ValueError("Dynamic similarity threshold requires 'initial_similarity_threshold' and "
                             "'final_similarity_threshold' in env_config.")
        if self.config.initial_similarity_threshold > self.config.final_similarity_threshold:
             print(f"Warning: initial_similarity_threshold ({self.config.initial_similarity_threshold}) > final_similarity_threshold ({self.config.final_similarity_threshold}). Threshold might decrease if total_training_steps=1.")
        # ---------------------------------------

        # --- Internal State ---
        self._original_base_molecule_smiles: Optional[str] = None
        self._current_molecule_smiles: Optional[str] = None
        self._original_mol_object = None # RDKit Mol object for original
        self._current_mol_object = None  # RDKit Mol object for current
        self._current_step: int = 0 # Episode step counter
        self._initial_score: float = 0.0 # Store score of the base molecule
        self._global_training_step: int = 0 # To be set during reset
        self._total_training_steps: int = 1 # To be set during reset

        # --- Rendering ---
        self.render_mode = self.config.render_mode
        self.render_cache: Optional[str] = None # Stores the text to be returned by render()

        # --- Initialize Oracle ---
        oracle_cfg = self.config.oracle_config
        # Use the determined single target_property for the main oracle
        multi_objective_weights = getattr(self.config, 'multi_objective_weights', None)
        self.oracle = MoleculeOracle(
            target_property=self.target_property,
            cache_file=oracle_cfg.get("cache_file"),
            multi_objective_weights=multi_objective_weights
        )

        # --- Track best molecule ---
        self._best_molecule_smiles: Optional[str] = None
        self._best_mol_object = None
        # Initialize best score based on optimization direction
        self._best_score: float = float('inf') if self.minimize_target else float('-inf')
        self._best_step: int = 0
        self._similarity_at_best_score: float = 1.0 # Similarity when the best score was achieved
        self._consecutive_failures: int = 0 # Add consecutive failure counter
        self._oracle_calls_this_episode: int = 0        # Track real oracle calls within an episode (no cache hits)
        self._total_oracle_calls_this_episode: int = 0  # Track total oracle calls within an episode (including cache hits)
        # Additional tracking for multi-objective tasks
        self._best_weighted_score: float = float('-inf')  # weighted total score
        self._best_individual_properties: Dict = {}  # best values for each property
        self._best_global_improvements: Dict = {}  # improvements relative to the original molecule
    
    def _parse_task_config(self):
        """Parse task configuration, generate task description and similarity threshold, and automatically set optimization direction"""
        try:
            # Get task configuration from env_config
            task = self.env_config.get('molecule_opt_task', 'qed')
            # similarity_threshold is still needed for debug print (now commented out), but still need to get for subsequent use
            similarity_threshold = self.env_config.get('molecule_opt_similarity_threshold', 0.4)
            
            # Validate task validity
            if not validate_task(task):
                raise ValueError(f"Invalid task: {task}. Task should be in format like 'qed', 'qed+logp', etc.")
            
            # Generate task description and similarity threshold string
            task_description, similarity_threshold_str = parse_task_config(self.env_config)
            
            # Store parsed results for subsequent variable replacement
            self.task_description = task_description
            self.similarity_threshold_str = similarity_threshold_str
            self.molecule_opt_task = task
            
            assert similarity_threshold is not None, "Similarity threshold should be properly set"
            
        except Exception as e:
            print(f"⚠️ Task parsing failed: {e}. Using default configuration.")
            self.task_description = "increase QED"
            self.similarity_threshold_str = "0.6"
            self.molecule_opt_task = "qed"
    
    def replace_instruction_variables(self, instruction: str) -> str:
        if instruction is None:
            return ""
        
        # Replace task description variable
        instruction = instruction.replace("${task_description}", self.task_description)
        # Replace similarity threshold variable
        instruction = instruction.replace("${similarity_threshold}", self.similarity_threshold_str)
        
        return instruction
    
    def _calculate_multi_objective_reward(self, old_smiles: str, new_smiles: str) -> tuple[float, dict]:
        """
        Calculate reward for multi-objective task
        
        Args:
            old_smiles: molecule SMILES before modification
            new_smiles: molecule SMILES after modification
            
        Returns:
            tuple: (reward, reward_info)
        """
        task_properties = self.molecule_opt_task.split('+')
        
        # If single-objective task, use original simple logic
        if len(task_properties) == 1:
            old_score = self.oracle(old_smiles)
            new_score = self.oracle(new_smiles)
            improvement = new_score - old_score
            
            if self.minimize_target:
                if improvement < 0:  # score decreases (good)
                    reward = -improvement * 5
                elif improvement > 0:  # score increases (bad)
                    reward = -improvement
                else:
                    reward = 0
            else:
                if improvement > 0:  # score increases (good)
                    reward = improvement * 5
                elif improvement < 0:  # score decreases (bad)
                    reward = improvement
                else:
                    reward = 0
            
            reward_info = {
                "improvement": improvement,
                "old_score": old_score,
                "new_score": new_score,
                "is_multi_objective": False
            }
            
            return reward, reward_info
        
        # Multi-objective task: evaluate task-related properties
        old_properties = self.oracle.evaluate_specific_properties(old_smiles, task_properties)
        new_properties = self.oracle.evaluate_specific_properties(new_smiles, task_properties)
        
        # Get original molecule properties (for calculating global improvements)
        original_properties = {}
        if hasattr(self, '_original_base_molecule_smiles'):
            original_properties = self.oracle.evaluate_specific_properties(
                self._original_base_molecule_smiles, task_properties)
        else:
            # If no original molecule information, use old_properties as fallback
            original_properties = old_properties
        
        # Calculate global improvements relative to the original molecule (this is the main reward basis)
        global_improvements = {}
        for prop in task_properties:
            original_value = original_properties.get(prop, 0.0)
            new_value = new_properties.get(prop, 0.0)
            global_improvements[prop] = new_value - original_value
        
        # Use global improvements to calculate reward
        total_reward, reward_info = calculate_multi_objective_reward(
            global_improvements, task_properties, self.optimization_directions)
        
        # Calculate weighted total score (for determining if it's the best molecule)
        weighted_score = 0.0
        from ragen.env.molecule_opt.property_utils import PROPERTY_DIFFICULTY_WEIGHTS
        for prop in task_properties:
            weight = PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0)
            # For minimize properties (like SA), use negative values
            if self.optimization_directions.get(prop, "maximize") == "minimize":
                weighted_score -= new_properties[prop] * weight
            else:
                weighted_score += new_properties[prop] * weight
        

        reward_info.update({
            "old_properties": old_properties,
            "new_properties": new_properties,
            "original_properties": original_properties,
            "global_improvements": global_improvements,
            "improvements": global_improvements,  # keep backward compatibility
            "weighted_score": weighted_score,  # for determining the best molecule
            "is_multi_objective": True,
            "task_properties": task_properties
        })
        
        return total_reward, reward_info

    # --- Core Gym Methods ---

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict]:
        """
        Resets the environment using an initial SMILES string provided via 'options'.
        Also receives global training step information via 'options'.
        Returns a *minimal* initial observation.
        """
        super().reset(seed=seed)
        options = options or {}

        # --- Store global step information ---
        self._global_training_step = options.get("global_step", 0)
        self._total_training_steps = options.get("total_training_steps", 1) # Default to 1 to avoid div by zero later
        if self._total_training_steps <= 0:
             print(f"Warning: Received total_training_steps={self._total_training_steps}. Setting to 1.")
             self._total_training_steps = 1
        # -------------------------------------

        initial_smiles_from_data = options.get("initial_smiles")
        if initial_smiles_from_data is None:
             print("CRITICAL WARNING: Initial SMILES not provided to MoleculeEnv.reset() via options. Using default 'CCO'. Data flow needs modification.")
             initial_smiles_from_data = "CCO"

        try:
            mol = Chem.MolFromSmiles(initial_smiles_from_data)
            if mol is None:
                raise ValueError(f"Invalid initial SMILES provided: {initial_smiles_from_data}")
            self._original_base_molecule_smiles = Chem.MolToSmiles(mol)
            self._original_mol_object = Chem.MolFromSmiles(self._original_base_molecule_smiles)
        except Exception as e:
             print(f"ERROR processing initial SMILES '{initial_smiles_from_data}': {e}. Resetting with 'CCO'.")
             self._original_base_molecule_smiles = "CCO"
             self._original_mol_object = Chem.MolFromSmiles("CCO")

        # Check if we should start from an elite molecule
        self._starting_from_elite = False
        self._elite_score = None
        self._elite_similarity = None
        self._elite_success_boost = 0.0  # Additional success threshold boost
        elite_molecule = options.get("elite_molecule")
        if elite_molecule:
            try:
                elite_mol = Chem.MolFromSmiles(elite_molecule['smiles'])
                if elite_mol is not None:
                    self._current_molecule_smiles = Chem.MolToSmiles(elite_mol)
                    self._current_mol_object = elite_mol
                    self._starting_from_elite = True
                    self._elite_score = elite_molecule.get('score', 0.0)
                    self._elite_similarity = elite_molecule.get('similarity', 0.0)
                    
                    elite_success_boost_config = options.get("elite_success_boost", 0.05)
                    self._elite_success_boost = elite_success_boost_config

                else:
                    print(f"WARNING: Invalid elite molecule SMILES: {elite_molecule['smiles']}. Starting from original.")
                    self._current_molecule_smiles = self._original_base_molecule_smiles
                    self._current_mol_object = self._original_mol_object
            except Exception as e:
                print(f"ERROR processing elite molecule: {e}. Starting from original.")
                self._current_molecule_smiles = self._original_base_molecule_smiles
                self._current_mol_object = self._original_mol_object
        else:
            self._current_molecule_smiles = self._original_base_molecule_smiles
            self._current_mol_object = self._original_mol_object
        self._current_step = 0 # Reset episode step counter
        self._initial_score = self.oracle(self._current_molecule_smiles)
        self._oracle_calls_this_episode = 0        # Initial molecule score does not count as real oracle calls
        self._total_oracle_calls_this_episode = 0  # Initial molecule score does not count as total oracle calls

        if self._starting_from_elite:
            self._best_molecule_smiles = self._current_molecule_smiles
            self._best_mol_object = self._current_mol_object
            self._best_score = self._elite_score
            self._best_step = 0
            self._similarity_at_best_score = self._elite_similarity
            
            task_properties = self.molecule_opt_task.split('+')
            if len(task_properties) > 1:
                # Calculate weighted score and properties of elite molecule
                try:
                    elite_properties = self.oracle.evaluate_specific_properties(self._current_molecule_smiles, task_properties)
                    self._best_individual_properties = elite_properties.copy()
                    
                    # Calculate improvements relative to the original molecule
                    original_properties = self.oracle.evaluate_specific_properties(self._original_base_molecule_smiles, task_properties)
                    global_improvements = {}
                    for prop in task_properties:
                        global_improvements[prop] = elite_properties[prop] - original_properties[prop]
                    self._best_global_improvements = global_improvements
                    
                    # Calculate weighted score
                    from ragen.env.molecule_opt.property_utils import PROPERTY_DIFFICULTY_WEIGHTS
                    weighted_score = 0.0
                    for prop in task_properties:
                        weight = PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0)
                        if self.optimization_directions.get(prop, "maximize") == "minimize":
                            weighted_score -= elite_properties[prop] * weight
                        else:
                            weighted_score += elite_properties[prop] * weight
                    self._best_weighted_score = weighted_score
                except Exception as e:
                    print(f"Warning: Failed to calculate elite weighted score: {e}")
                    self._best_weighted_score = float('-inf')
                    self._best_individual_properties = {}
                    self._best_global_improvements = {}
            else:
                self._best_weighted_score = float('-inf')
                self._best_individual_properties = {}
                self._best_global_improvements = {}
        else:
            self._best_molecule_smiles = self._original_base_molecule_smiles
            self._best_mol_object = self._original_mol_object
            self._best_score = self._initial_score
            self._best_step = 0
            self._similarity_at_best_score = 1.0
            # Multi-objective task reset
            self._best_weighted_score = float('-inf')
            self._best_individual_properties = {}
            self._best_global_improvements = {}
        self._consecutive_failures = 0

        # Get task-related properties list
        task_properties = self.molecule_opt_task.split('+')
        
        # Get original molecule task-related properties scores (for multi-objective display)
        original_all_scores = self.oracle.evaluate_specific_properties(self._original_base_molecule_smiles, task_properties)
        
        # Generate observation based on whether we're starting from elite
        if self._starting_from_elite:
            # Get elite molecule task-related properties scores
            elite_all_scores = self.oracle.evaluate_specific_properties(self._current_molecule_smiles, task_properties)
            
            # Generate initial properties display
            initial_guidance = generate_complete_multi_objective_guidance(
                original_scores={}, 
                new_scores=original_all_scores, 
                task_properties=task_properties,
                reward_info={},
                is_initial=True
            )
            
            # Generate elite molecule properties display  
            elite_guidance = generate_complete_multi_objective_guidance(
                original_scores=original_all_scores,
                new_scores=elite_all_scores,
                task_properties=task_properties,
                reward_info={},
                is_initial=False
            )
            
            observation = (
                f"Original Molecule: {self._original_base_molecule_smiles}\n"
                f"{initial_guidance}\n"
                f"Starting from Elite Molecule: {self._current_molecule_smiles}\n"
                f"{elite_guidance}\n"
                f"Elite Similarity to Original: {self._elite_similarity:.3f}\n"
                f"Note: You are starting from an already optimized molecule. "
                f"Continue optimizing while maintaining similarity ≥ {self.config.final_similarity_threshold} with the ORIGINAL molecule."
            )
        else:
            # Generate initial properties display
            initial_guidance = generate_complete_multi_objective_guidance(
                original_scores={}, 
                new_scores=original_all_scores, 
                task_properties=task_properties,
                reward_info={},
                is_initial=True
            )
            
            observation = f"Original Molecule: {self._original_base_molecule_smiles}\n{initial_guidance}"
        self.render_cache = observation

        info = {"initial_smiles": self._original_base_molecule_smiles, "initial_score": self._initial_score}
        return observation, info

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """Apply molecule modification"""
        terminated = False
        truncated = False
        reward = 0.0
        is_step_successful = False

        score_before_modification = self.oracle(self._current_molecule_smiles)

        current_similarity_threshold = self.config.final_similarity_threshold

        info = {
            "action_received": action,
            "similarity": 1.0,
            "score_before": score_before_modification,
            "score_after": score_before_modification,
            "is_valid_modification": False,
            "is_similar_enough": True,
            "final_smiles": self._current_molecule_smiles,
            "current_similarity_threshold": current_similarity_threshold, # Log the threshold used
            "oracle_calls_this_step": 0,  # Track real oracle calls in this step
            "total_oracle_calls_this_step": 0,  # Track total oracle calls in this step (including cache)
            "oracle_calls_this_episode": self._oracle_calls_this_episode, 
            "total_oracle_calls_this_episode": self._total_oracle_calls_this_episode,  # Track episode cumulative total oracle calls
            "current_best_score": self._best_score,  # Current best score
            "current_best_smiles": self._best_molecule_smiles  # Current best molecule
        }

        try:
            next_mol_object = Chem.MolFromSmiles(action)
            if next_mol_object is None:
                is_step_successful = False
                reward = -0.5
                info["is_valid_modification"] = False
            else:
                next_molecule_smiles = Chem.MolToSmiles(next_mol_object, canonical=True)
                current_canonical_smiles = Chem.MolToSmiles(self._current_mol_object, canonical=True)

                if next_molecule_smiles == current_canonical_smiles:
                    is_step_successful = False
                    reward = -0.3
                    info["is_valid_modification"] = True
                    info["no_change_detected_in_step"] = True
                else:
                    similarity = self._calculate_similarity(next_mol_object, self._original_mol_object)
                    info["similarity"] = similarity

                    if similarity < current_similarity_threshold:
                        is_step_successful = False
                        info["is_similar_enough"] = False
                        reward = -2 * (current_similarity_threshold - similarity)
                        info["is_valid_modification"] = True
                    else: # Passed similarity check
                        # Check structural constraints (heavy atom count, carbon chain length, etc.)
                        constraint_valid, constraint_message = self._check_structural_constraints(next_mol_object)
                        if not constraint_valid:
                            is_step_successful = False
                            reward = -1.0  # Penalty for violating structural constraints
                            info.update({
                                "is_valid_modification": True,
                                "is_similar_enough": True,
                                "violates_structural_constraints": True,
                                "_constraint_violation_message": constraint_message  # Use underscore prefix to avoid metric logging
                            })
                        else: # Passed both similarity and structural constraints
                            next_score, was_cached = self.oracle.evaluate_with_cache_info(next_molecule_smiles)
                            
                            # Total oracle calls (including cache hits)
                            self._total_oracle_calls_this_episode += 1
                            info["total_oracle_calls_this_step"] = 1
                            
                            # Only non-cached calls count as real oracle calls
                            if not was_cached:
                                self._oracle_calls_this_episode += 1
                                info["oracle_calls_this_step"] = 1 
                            else:
                                info["oracle_calls_this_step"] = 0 

                            improvement = next_score - score_before_modification
                            info.update({
                                "score_after": next_score,
                                "is_valid_modification": True,
                                "is_similar_enough": True,
                                "violates_structural_constraints": False,
                                "oracle_calls_this_episode": self._oracle_calls_this_episode,  # Update cumulative real oracle calls
                                "total_oracle_calls_this_episode": self._total_oracle_calls_this_episode  # Update cumulative total oracle calls
                            })

                        is_step_successful = True # Assume success initially if valid & similar

                        # --- Check if multi-objective task ---
                        task_properties = self.molecule_opt_task.split('+')
                        if len(task_properties) > 1:
                            # Multi-objective task: use multi-objective reward calculation
                            multi_reward, multi_reward_info = self._calculate_multi_objective_reward(
                                self._current_molecule_smiles, next_molecule_smiles
                            )
                            reward = multi_reward
                            
                            # Use weighted score to determine if it's the best molecule
                            weighted_score = multi_reward_info.get('weighted_score', 0.0)
                            current_best_weighted_score = getattr(self, '_best_weighted_score', float('-inf'))
                            
                            if weighted_score > current_best_weighted_score:
                                # Update best molecule (record all property information)
                                self._best_molecule_smiles = next_molecule_smiles
                                self._best_mol_object = next_mol_object
                                self._best_weighted_score = weighted_score
                                self._best_individual_properties = multi_reward_info.get('new_properties', {})
                                self._best_global_improvements = multi_reward_info.get('global_improvements', {})
                                self._best_score = weighted_score  # Use weighted_score as best_score
                                self._best_step = self._current_step
                                self._similarity_at_best_score = similarity
                                self._best_multi_reward = multi_reward
                                
                                info["is_new_best"] = True
                                info["current_best_score"] = self._best_weighted_score
                                info["current_best_smiles"] = self._best_molecule_smiles
                                info["current_best_properties"] = self._best_individual_properties
                                info["current_best_improvements"] = self._best_global_improvements
                            else:
                                info["is_new_best"] = False
                            
                            # Determine if the step is successful based on global improvements
                            if multi_reward < 0:
                                is_step_successful = False
                            
                            # Add multi-objective information to info
                            info["multi_objective_info"] = multi_reward_info
                        
                        # --- Single-objective Conditional Reward and Best Score Logic ---
                        elif self.minimize_target: # Minimization Logic (e.g., SA Score)
                            if improvement < 0: # Score decreased (good)
                                reward = -improvement * 5 # Positive reward for decrease
                                if next_score < self._best_score:
                                    self._best_molecule_smiles = next_molecule_smiles
                                    self._best_mol_object = next_mol_object
                                    self._best_score = next_score
                                    self._best_step = self._current_step
                                    self._similarity_at_best_score = similarity
                                    info["is_new_best"] = True
                                    # Update best results in info
                                    info["current_best_score"] = self._best_score
                                    info["current_best_smiles"] = self._best_molecule_smiles
                                else:
                                    info["is_new_best"] = False
                            elif improvement > 0: # Score increased (bad)
                                is_step_successful = False # Not a successful improvement step
                                reward = -improvement # Negative reward
                                info["is_new_best"] = False
                            else: # improvement == 0
                                reward = 0
                                info["is_new_best"] = False
                                # is_step_successful remains True
                        else: # Maximization Logic (default)
                            if improvement > 0: # Score increased (good)
                                reward = improvement * 5 # Positive reward for increase
                                if next_score > self._best_score:
                                    self._best_molecule_smiles = next_molecule_smiles
                                    self._best_mol_object = next_mol_object
                                    self._best_score = next_score
                                    self._best_step = self._current_step
                                    self._similarity_at_best_score = similarity
                                    info["is_new_best"] = True
                                    # Update best results in info
                                    info["current_best_score"] = self._best_score
                                    info["current_best_smiles"] = self._best_molecule_smiles
                                else:
                                    info["is_new_best"] = False
                            elif improvement < 0: # Score decreased (bad)
                                is_step_successful = False # Not a successful improvement step
                                reward = improvement # Negative reward (already negative)
                                info["is_new_best"] = False
                            else: # improvement == 0
                                reward = 0
                                info["is_new_best"] = False
                                # is_step_successful remains True
                        # --- End Conditional Logic ---

                        # --- Check for QED success condition (overrides) ---
                        # Only apply success termination condition for single-objective QED task
                        task_properties = self.molecule_opt_task.split('+')
                        is_single_qed_task = (len(task_properties) == 1 and self.target_property == "qed")
                        
                        if is_single_qed_task:
                            # Dynamic success threshold: if starting from elite molecule, need higher score
                            qed_success_threshold = 0.9 + self._elite_success_boost
                            if next_score > qed_success_threshold:
                                # This is a success regardless of immediate improvement, 
                                # as long as SMILES was valid and similarity met.
                                terminated = True
                                truncated = False # Ensure this success isn't seen as truncation
                                info["success"] = True
                                is_step_successful = True # Crucial: mark this step as ultimately successful
                                
                                # Override reward for achieving QED > threshold success
                                # Using a significant positive value.
                                reward = 5.0 

                        # Update current molecule only if the step was deemed successful (or no change)
                        if is_step_successful or improvement == 0:
                            self._current_molecule_smiles = next_molecule_smiles
                            self._current_mol_object = next_mol_object
                            info["final_smiles"] = self._current_molecule_smiles

        except Exception as e:
            is_step_successful = False
            reward = -1
            info["is_valid_modification"] = False
            info["_error"] = str(e)  # Use underscore prefix to avoid metric logging

        # Distinguish failure types: only invalid modification, similarity not enough, or structural constraint violation count as true failure
        if not is_step_successful:
            # Check if it's a serious failure (invalid SMILES, similarity not enough, no change, structural constraint violation)
            if (not info["is_valid_modification"] or 
                not info["is_similar_enough"] or 
                info.get("no_change_detected_in_step", False) or
                info.get("violates_structural_constraints", False)):
                self._consecutive_failures += 1
                if self._consecutive_failures >= 1 and self._best_molecule_smiles is not None and self._best_molecule_smiles != self._current_molecule_smiles:
                    self._current_molecule_smiles = self._best_molecule_smiles
                    self._current_mol_object = self._best_mol_object
                    info["rollback_to_best"] = True
                    info["rollback_triggered_by_consecutive_failures"] = True
                    info["final_smiles"] = self._current_molecule_smiles
                    self._consecutive_failures = 0
            # If valid modification but no score improvement, don't count as failure, reset counter
            else:
                self._consecutive_failures = 0
        else:
            self._consecutive_failures = 0

        # Update EPISODE step counter and check EPISODE termination
        self._current_step += 1
        if self._current_step >= self.config.max_steps:
            truncated = True

        observation = self._get_step_observation(reward, info)
        self.render_cache = observation
        done = terminated or truncated


        return observation, reward, done, info

    def render(self) -> Optional[str]:
        """Returns the cached textual representation of the current state."""
        if self.render_mode == "text":
            return self.render_cache
        else:
             raise NotImplementedError(f"Render mode '{self.render_mode}' not supported.")

    def close(self):
        """Clean up any resources."""
        if hasattr(self.oracle, 'save_cache') and self.config.oracle_config.get('cache_file'):
             try:
                 self.oracle.save_cache(self.config.oracle_config['cache_file'])
                 print(f"Oracle cache saved to {self.config.oracle_config['cache_file']}")
             except Exception as e:
                 print(f"Error saving oracle cache: {e}")
        self.render_cache = None
        print("MoleculeEnv closed.")

    # --- Helper Methods ---

    def _get_current_similarity_threshold(self) -> float:
        """
        Calculates the similarity threshold based on the GLOBAL training step.
        Uses a cosine curve, reaching the final threshold at 75% of total steps.
        """
        initial_thresh = self.config.initial_similarity_threshold
        final_thresh = self.config.final_similarity_threshold
        total_steps = self._total_training_steps
        current_step_index = self._global_training_step # Use global step

        if total_steps <= 1:
            return final_thresh

        # Calculate the linear fraction of training completed (0.0 to 1.0)
        clamped_step_index = min(current_step_index, total_steps - 1)
        linear_fraction = clamped_step_index / (total_steps - 1) if total_steps > 1 else 1.0

        # Scale the fraction so it reaches 1.0 at 75% of training
        target_fraction = 0.75
        if target_fraction <= 0:
            scaled_fraction = 1.0 # Avoid division by zero, go straight to final
        else:
            scaled_fraction = linear_fraction / target_fraction

        # Clamp the scaled fraction to a maximum of 1.0
        clamped_scaled_fraction = min(scaled_fraction, 1.0)

        # Apply cosine transformation using the clamped scaled fraction
        cosine_fraction = (1 - math.cos(math.pi * clamped_scaled_fraction)) / 2.0

        # Interpolate using the cosine fraction
        current_threshold = initial_thresh + cosine_fraction * (final_thresh - initial_thresh)

        # Clamp the result to handle potential floating point issues or reversed initial/final
        min_thresh = min(initial_thresh, final_thresh)
        max_thresh = max(initial_thresh, final_thresh)
        current_threshold = max(min(current_threshold, max_thresh), min_thresh)

        return current_threshold

    def _get_step_observation(self, last_reward: float, last_info: Dict) -> str:
        # Use current_step (episode step) + 1 for 1-based display
        step_info = f"Step {self._current_step} of {self.config.max_steps}\\n"
        target_prop_name = PROPERTY_MAPPING.get(self.target_property, {}).get('description', self.target_property)
        direction_hint = "(lower is better)" if self.minimize_target else "(higher is better)"
        
        # Helper function for detailed score display in multi-objective optimization
        def get_detailed_score_display(smiles, score, label="Score"):
            if self.target_property == 'logp_qed':
                components = self.oracle.evaluate_with_components(smiles)
                return (f"{label} (LogP: {components['logp']:.3f}, QED: {components['qed']:.3f}, "
                       f"Combined: {components['combined']:.3f})")
            else:
                return f"{label}: {score:.3f}"

        rollback_message_segment = "" # Stores the detailed message if a rollback happened
        if last_info.get("rollback_triggered_by_consecutive_failures", False):
            failed_attempt_smiles = last_info.get('action_received', 'N/A')
            score_of_failed_attempt = last_info.get('score_after', self.oracle(self._current_molecule_smiles)) 
            score_before_failed_attempt = last_info.get('score_before', self.oracle(self._current_molecule_smiles))
            failed_improvement = score_of_failed_attempt - score_before_failed_attempt

            failed_attempt_details = (
                f"Your last proposed modification was to '{failed_attempt_smiles}'.\\n"
                f"This resulted in a score of {score_of_failed_attempt:.3f} {direction_hint} (change from {score_before_failed_attempt:.3f}: {failed_improvement:+.3f}).\\n"
            )
            rollback_message_segment = (
                 f"Due to the unsuccessful modification detailed above, the environment has reverted to your previously best molecule (from step {self._best_step + 1}).\\n"
                 f"{failed_attempt_details}"
                 f"The best score achieved so far is {self._best_score:.3f} {direction_hint} with SMILES: {self._best_molecule_smiles}.\\n"
                 f"Your current molecule for the next attempt is now: {self._current_molecule_smiles} (Score: {self.oracle(self._current_molecule_smiles):.3f} {direction_hint}).\\n"
                 f"Please consider different modification strategies.\\n"
            )
        
        current_required_threshold = last_info.get("current_similarity_threshold", self.config.final_similarity_threshold)
        # This is the molecule that the LLM will ACTUALLY be working on for the NEXT turn.
        # After a rollback, this IS the best molecule.
        current_mol_for_next_turn_display = self._current_molecule_smiles
        score_of_current_mol_for_next_turn = self.oracle(current_mol_for_next_turn_display)

        # --- Construct the main message based on the outcome of the last action ---

        # Case 1: Rollback occurred. The rollback_message_segment is primary.
        if rollback_message_segment:
            return (
                f"{step_info}"
                f"{rollback_message_segment}" # This contains all necessary info about the rollback and current state
                f"Reward for the last attempted action (that triggered rollback): {last_reward:.3f}\\n"
            )

        # Case 2: No rollback, but other issues with the last action.
        # These messages will directly follow step_info.
        if not last_info["is_valid_modification"]:
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                # rollback_message_segment is empty here
                f"Invalid SMILES format provided: '{last_info.get('action_received', 'N/A')}'. Ensure the SMILES is correct.\\n"
                f"Current molecule remains:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Current required similarity >= {current_required_threshold:.3f}\\n"
                f"Reward: {last_reward:.3f}\\n"
            )
        
        if last_info.get("no_change_detected_in_step", False):
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                # rollback_message_segment is empty here
                f"No modification detected. Your proposed SMILES '{last_info.get('action_received')}' matches the current molecule.\\n"
                f"Current molecule remains:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Current required similarity >= {current_required_threshold:.3f}\\n"
                f"Please propose a different modification.\\n" # Reward for no change is often 0 or small penalty
                f"Reward: {last_reward:.3f}\\n"

            )

        if not last_info["is_similar_enough"]:
            similarity_achieved = last_info.get('similarity', 0.0)
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                # rollback_message_segment is empty here
                f"Similarity too low: {similarity_achieved:.3f} < required {current_required_threshold:.3f} for proposed SMILES '{last_info.get('action_received', 'N/A')}'.\\n"
                f"Current molecule remains:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Reward: {last_reward:.3f}\\n"
                f"Consider smaller, more conservative changes to the current molecule.\\n"
            )
        
        if last_info.get("violates_structural_constraints", False):
            constraint_message = last_info.get("_constraint_violation_message", "Unknown constraint violation")
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                # rollback_message_segment is empty here
                f"Structural constraint violation: {constraint_message}\\n"
                f"Proposed SMILES: '{last_info.get('action_received', 'N/A')}'\\n"
                f"Current molecule remains unchanged:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Please propose more reasonable molecular modifications. Avoid simply adding carbon atoms or creating overly large molecules.\\n"
                f"Reward: {last_reward:.3f}\\n"
            )

        # Case 3: No rollback, valid, similar, and different modification. This is a "normal" step outcome.
        # This part will only be reached if rollback_message_segment is empty AND none of the above conditions were met.
        actual_modified_molecule = last_info.get("final_smiles", current_mol_for_next_turn_display) # Should be the successfully modified one
        score_of_modified_molecule = last_info['score_after']
        improvement = score_of_modified_molecule - last_info["score_before"]
        similarity_achieved = last_info.get('similarity', 0.0)
        # Generate guidance information and property score display
        task_properties = self.molecule_opt_task.split('+')
        multi_obj_info = last_info.get("multi_objective_info", {})
        
        if multi_obj_info.get("is_multi_objective", False) or len(task_properties) > 1:
            # Multi-objective task: use complete guidance generation (includes score display)
            old_properties = multi_obj_info.get("old_properties", {})
            new_properties = multi_obj_info.get("new_properties", {})
            
            guidance = generate_complete_multi_objective_guidance(
                original_scores=old_properties,
                new_scores=new_properties,
                task_properties=task_properties,
                reward_info=multi_obj_info,
                is_initial=False
            )
        else:
            # Single-objective task: use original logic
            guidance = ""
            if self.minimize_target:
                if last_info.get("is_new_best", False):
                    guidance = f"Excellent! New best score achieved {direction_hint}! Keep improving."
                elif improvement < 0: # Good for minimization
                    guidance = f"Good improvement! Score decreased {direction_hint}. Continue reducing the score."
                elif improvement == 0:
                    guidance = "No score change. Valid modification, but property didn't change. Try alternative strategies."
                else: # improvement > 0 (Bad for minimization)
                    guidance = f"Score increased {direction_hint}. This worsened the property. Try different modifications to reduce the score."
            else: # Maximization
                if last_info.get("is_new_best", False):
                    guidance = f"Great job! New best score achieved {direction_hint}! Keep refining."
                elif improvement > 0: # Good for maximization
                    guidance = f"Good improvement! Score increased {direction_hint}. Keep refining."
                elif improvement == 0:
                    guidance = "No score change. Valid modification, but property didn't change. Try alternative strategies."
                else: # improvement < 0 (Bad for maximization)
                    guidance = f"Score decreased. Valid modification, but negatively impacted property {direction_hint}. Consider alternative strategies."
        
        # Get the molecule SMILES for before score
        before_smiles = last_info.get('action_received', current_mol_for_next_turn_display)  # This is the molecule BEFORE modification
        before_score_display = get_detailed_score_display(
            self._current_molecule_smiles if hasattr(self, '_current_molecule_smiles') else before_smiles, 
            last_info['score_before'], 
            "Previous Score"
        )
        
        # Get after score display
        after_score_display = get_detailed_score_display(
            actual_modified_molecule, 
            score_of_modified_molecule, 
            f"New Score ({target_prop_name} {direction_hint})"
        )
        
        return (
            f"{step_info}"
            # rollback_message_segment is empty here
            f"Action: Proposed '{last_info.get('action_received', 'N/A')}'\\n"
            f"{before_score_display}\\n"
            f"Resulting Molecule: {actual_modified_molecule}\\n"
            f"Similarity to Original: {similarity_achieved:.3f} (required >= {current_required_threshold:.3f})\\n"
            f"{after_score_display} (change: {improvement:+.3f})\\n"
            f"{guidance}\\n"
            f"Reward: {last_reward:.3f}\\n"
        )

    def _calculate_similarity(self, mol1: Optional[Chem.Mol], mol2: Optional[Chem.Mol]) -> float:
        """Calculates Tanimoto similarity between two RDKit Mol objects."""
        if mol1 is None or mol2 is None:
            return 0.0
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception as e:
            print(f"Warning: Similarity calculation failed - {e}")
            return 0.0

    def _get_heavy_atom_count(self, mol: Optional[Chem.Mol]) -> int:
        """Calculate the number of heavy atoms (non-hydrogen atoms) in the molecule"""
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()

    def _get_longest_carbon_chain(self, mol: Optional[Chem.Mol]) -> int:
        """Calculate the longest linear carbon chain length in the molecule"""
        if mol is None:
            return 0
        
        try:
            # Find all carbon atoms
            carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
            
            if not carbon_atoms:
                return 0
            
            max_chain_length = 0
            
            # Perform depth-first search for each carbon atom to find the longest carbon chain
            for start_carbon in carbon_atoms:
                visited = set()
                chain_length = self._dfs_carbon_chain(mol, start_carbon, visited)
                max_chain_length = max(max_chain_length, chain_length)
            
            return max_chain_length
        except Exception as e:
            print(f"Warning: Carbon chain calculation failed - {e}")
            return 0

    def _dfs_carbon_chain(self, mol: Chem.Mol, atom_idx: int, visited: set) -> int:
        """Depth-first search to calculate the longest carbon chain starting from a specified carbon atom"""
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Ensure current atom is a carbon atom
        if atom.GetSymbol() != 'C':
            return 0
        
        max_length = 1  # Current carbon atom counts as 1
        
        # Traverse all neighboring carbon atoms
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor.GetSymbol() == 'C' and neighbor_idx not in visited:
                # Recursively calculate chain length starting from neighboring carbon atom
                chain_length = 1 + self._dfs_carbon_chain(mol, neighbor_idx, visited.copy())
                max_length = max(max_length, chain_length)
        
        return max_length

    def _check_structural_constraints(self, mol: Optional[Chem.Mol]) -> tuple[bool, str]:
        """
        Check if molecule violates structural constraints
        
        Returns:
            (is_valid, violation_message)
        """
        if mol is None:
            return False, "Invalid molecule structure"
        
        # Check heavy atom count
        heavy_atom_count = self._get_heavy_atom_count(mol)
        max_heavy_atoms = self.config.max_heavy_atoms
        if heavy_atom_count > max_heavy_atoms:
            return False, f"Molecule too large: contains {heavy_atom_count} heavy atoms (limit ≤{max_heavy_atoms}). Large molecules typically have poor synthesizability and drug-likeness."
        
        # Check longest carbon chain
        longest_carbon_chain = self._get_longest_carbon_chain(mol)
        max_carbon_chain = self.config.max_carbon_chain_length
        if longest_carbon_chain > max_carbon_chain:
            return False, f"Carbon chain too long: longest linear carbon chain is {longest_carbon_chain} atoms (limit ≤{max_carbon_chain}). Long carbon chains reduce synthesizability and drug-likeness. Cannot simply add carbon atoms to optimize properties."
        
        return True, ""
