from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class MoleculeEnvConfig:
    """Configuration for the Molecule Optimization Environment."""

    # --- Task Specific Parameters ---
    # similarity_threshold: float = 0.4  # Remove old fixed threshold
    initial_similarity_threshold: float = 0.4 # Set to fixed 0.4 to match envs.yaml
    final_similarity_threshold: float = 0.4   # Set to fixed 0.4 to match envs.yaml
    max_steps: int = 5  # Maximum number of modification steps per episode
    minimize_property: bool = False # <<< Add flag to control optimization direction (False=Maximize, True=Minimize)
    
    # --- Structural Constraints ---
    max_heavy_atoms: int = 50      # Maximum number of heavy atoms (non-hydrogen)
    max_carbon_chain_length: int = 7  # Maximum length of linear carbon chain
    
    # --- Multi-objective Optimization ---
    multi_objective_weights: Dict = field(default_factory=lambda: {
        'logp': 0.5,  # Weight for LogP component in multi-objective optimization
        'qed': 0.5    # Weight for QED component in multi-objective optimization
    })

    # --- Oracle Configuration ---
    oracle_config: Dict = field(default_factory=lambda: {
        "target_property": "qed",  # This will be effectively overridden by the main target_property
        "cache_file": None  # Optional path to save/load oracle prediction cache
    })

    # --- Standard Env Parameters ---
    render_mode: str = "text"  # Only text rendering is supported currently
    target_property: str = "qed"  # <--- Add target property field with default

    # You might not need a discrete action lookup if actions are modifications described in text
    # action_lookup: Optional[Dict[int, str]] = None

    def __post_init__(self):
        # Validate the main target_property
        self.supported_properties = [
            "qed", "logp", "sa", "jnk3", "gsk3b", "drd2", "penalized_logp", "mr", "logp_qed" # Added penalized_logp, mr, and logp_qed
        ]
        if self.target_property not in self.supported_properties:
            raise ValueError(f"Unsupported target_property: {self.target_property}. Supported: {self.supported_properties}")
        
        # Optionally, ensure oracle_config uses the main target_property if needed later
        # Or just let the MoleculeOracle class handle the target_property passed to its init
        # if self.oracle_config.get("target_property") is None:
        #    self.oracle_config["target_property"] = self.target_property
        pass


@dataclass
class StaticExemplarMemoryConfig:
    """
    Configuration for Static Exemplar Memory (retrieval).

    Static Exemplar Memory provides cold-start grounding by retrieving similar
    molecules from a large pre-indexed chemical database.
    """
    enabled: bool = True
    url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 5
    timeout: float = 5.0
    trigger_mode: str = "on_stuck"  # "always", "on_stuck", "never"
    similarity_threshold: float = 0.4


@dataclass
class EvolvingSkillMemoryConfig:
    """
    Configuration for Evolving Skill Memory.

    Evolving Skill Memory distills successful optimization trajectories into
    reusable strategies that can be applied to future optimization tasks.
    """
    enabled: bool = True
    max_size: int = 10000
    save_path: str = "skill_memory.pkl"
    min_score_delta: float = 0.01
