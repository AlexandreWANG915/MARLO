import os
import numpy as np
import yaml
from rdkit import Chem
import tdc
import math
import shutil
import tempfile
import uuid
import time
import warnings
import logging

# Set TDC logging level to WARNING to reduce information output
logging.getLogger('tdc').setLevel(logging.WARNING)

# Global silent mode control
ORACLE_SILENT_MODE = os.environ.get('ORACLE_SILENT_MODE', 'false').lower() == 'true'

# Global TDC Oracle instance cache to avoid repeated creation (reduces "Found local copy" messages)
_GLOBAL_TDC_ORACLES = {}

def get_or_create_tdc_oracle(oracle_name: str):
    """
    Get or create TDC Oracle instance (global singleton pattern)
    
    Args:
        oracle_name: TDC Oracle name, such as 'QED', 'LogP', 'JNK3', etc.
    
    Returns:
        TDC Oracle instance
    """
    if oracle_name not in _GLOBAL_TDC_ORACLES:
        try:
            if not ORACLE_SILENT_MODE:
                print(f"🔄 Creating {oracle_name} evaluator for the first time...")
            # Temporarily suppress TDC output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _GLOBAL_TDC_ORACLES[oracle_name] = tdc.Oracle(name=oracle_name)
            if not ORACLE_SILENT_MODE:
                print(f"✅ {oracle_name} evaluator created successfully")
        except Exception as e:
            print(f"❌ Failed to create {oracle_name} evaluator: {e}")
            raise e
    return _GLOBAL_TDC_ORACLES[oracle_name]

def clear_global_tdc_oracles():
    """Clear global TDC Oracle cache (for memory management or re-initialization)"""
    global _GLOBAL_TDC_ORACLES
    _GLOBAL_TDC_ORACLES.clear()
    if not ORACLE_SILENT_MODE:
        print("🧹 Global TDC Oracle cache cleared")

def get_global_tdc_oracle_stats():
    """Get global TDC Oracle cache statistics"""
    return {
        'cached_oracles': list(_GLOBAL_TDC_ORACLES.keys()),
        'total_count': len(_GLOBAL_TDC_ORACLES)
    }

class MoleculeOracle:
    """Molecular property calculator"""
    
    def __init__(self, target_property='qed', cache_file=None, multi_objective_weights=None):
        self.target_property = target_property
        self.calls = 0                # Real Oracle call count (excluding cache hits)
        self.total_calls = 0          # Total call count (including cache hits)
        self.cache = {}
        self.use_fallback = False
        
        # Multi-objective optimization weights
        self.multi_objective_weights = multi_objective_weights or {'logp': 0.5, 'qed': 0.5}
        
        # Generate unique identifier for each Oracle instance to ensure process-level cache isolation
        self.instance_id = f"{os.getpid()}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        
        # Cache multi-property evaluators to avoid repeated creation (reduces "Found local copy" messages)
        self.property_evaluators = {}
        
        # Preload common evaluators to reduce runtime "Found local copy" messages
        self._preload_common_evaluators()
        
        # Initialize TDC evaluator (using global singleton)
        try:
            property_to_oracle = {
                'qed': 'QED',
                'logp': 'LogP',
                'jnk3': 'JNK3',
                'gsk3b': 'GSK3B',
                'drd2': 'DRD2',
                'sa': 'SA'
            }
            
            if target_property in property_to_oracle:
                oracle_name = property_to_oracle[target_property]
                try:
                    self.evaluator = get_or_create_tdc_oracle(oracle_name)
                except Exception as e:
                    if target_property == 'jnk3':
                        print(f"JNK3 model initial loading failed: {e}")
                        if not ORACLE_SILENT_MODE:
                            print("Attempting to clear TDC cache and reload...")
                        self._clear_tdc_cache()
                        # Retry loading
                        self.evaluator = get_or_create_tdc_oracle(oracle_name)
                        if not ORACLE_SILENT_MODE:
                            print(f"JNK3 model reloaded successfully")
                    else:
                        raise e
            elif target_property == 'mr':
                # MR (Molecular Refractivity) uses RDKit direct calculation
                self.use_fallback = True
                self.evaluator = None
            else:
                raise ValueError(f"Unsupported property: {target_property}")
        except Exception as e:
            if target_property in ['jnk3', 'gsk3b', 'drd2']:
                # For these bioactivity properties, throw error if loading fails, do not use fallback
                print(f"{target_property.upper()} model final loading failed: {e}")
                raise RuntimeError(f"{target_property.upper()} model loading failed: {e}. Please check network connection or manually clear TDC cache and retry.")
            else:
                raise
        
        # Load from cache file
        if cache_file and os.path.exists(cache_file):
            self.load_cache(cache_file)
    
    def _clear_tdc_cache(self):
        """Clear TDC cache directory"""
        import tdc
        try:
            # Get TDC data directory
            tdc_home = os.path.expanduser("~/.tdc")
            if os.path.exists(tdc_home):
                if not ORACLE_SILENT_MODE:
                    print(f"Clearing TDC cache directory: {tdc_home}")
                shutil.rmtree(tdc_home)
                if not ORACLE_SILENT_MODE:
                    print("TDC cache cleared")
            else:
                if not ORACLE_SILENT_MODE:
                    print("TDC cache directory not found")
        except Exception as e:
            print(f"Error occurred while clearing TDC cache: {e}")  # Always print error messages
    
    def _preload_common_evaluators(self):
        """Preload common molecular property evaluators (using global singleton) to reduce runtime "Found local copy" messages"""
        try:
            common_properties = {
                'qed': 'QED',
                'logp': 'LogP', 
                'sa': 'SA',
                'jnk3': 'JNK3',
                'gsk3b': 'GSK3B',
                'drd2': 'DRD2'
            }
            if not ORACLE_SILENT_MODE:
                print("🔄 Preloading molecular property evaluators (global singleton mode)...")
            
            for prop, oracle_name in common_properties.items():
                try:
                    # Use global singleton to get TDC Oracle instance
                    self.property_evaluators[prop] = get_or_create_tdc_oracle(oracle_name)
                except Exception as e:
                    if not ORACLE_SILENT_MODE:
                        print(f"⚠️  Failed to preload {prop} evaluator: {e}")
                    # Do not block initialization, try creating at runtime
                    continue
            
            if not ORACLE_SILENT_MODE:
                print(f"✅ Preloading completed, {len(self.property_evaluators)} evaluators in total")
        except Exception as e:
            if not ORACLE_SILENT_MODE:
                print(f"⚠️  Error occurred during evaluator preloading: {e}")
            # If preloading fails, maintain the original lazy loading mechanism
            self.property_evaluators = {}
    
    def __call__(self, smiles):
        """Calculate molecular property value"""
        if isinstance(smiles, list):
            return [self.evaluate(smi) for smi in smiles]
        else:
            return self.evaluate(smiles)
    
    def evaluate(self, smiles):
        """Evaluate single molecule"""
        if not smiles:
            return 0.0
        
        # Total call count (including cache hits)
        self.total_calls += 1
        
        # Check cache
        was_cached = smiles in self.cache
        if was_cached:
            return self.cache[smiles]
        
        # Calculate property (new call)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # If using fallback implementation
            if self.use_fallback:
                score = self._fallback_evaluate(smiles, mol)
            else:
                score = float(self.evaluator(smiles))
            
            # Handle invalid values
            if math.isnan(score):
                score = 0.0
            
            # Update cache and real call count (only new calls are counted)
            self.cache[smiles] = score
            self.calls += 1
            
            return score
        except Exception as e:
            print(f"Property calculation error: {e}")
            return 0.0
    
    def evaluate_with_cache_info(self, smiles):
        """Evaluate molecule and return whether it was a cached call"""
        if not smiles:
            return 0.0, True
        
        # Total call count (including cache hits)
        self.total_calls += 1
        
        # Check cache
        was_cached = smiles in self.cache
        if was_cached:
            return self.cache[smiles], True
        
        # Calculate property (new call)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0, False
            
            # If using fallback implementation
            if self.use_fallback:
                score = self._fallback_evaluate(smiles, mol)
            else:
                score = float(self.evaluator(smiles))
            
            # Handle invalid values
            if math.isnan(score):
                score = 0.0
            
            # Update cache and real call count (only new calls are counted)
            self.cache[smiles] = score
            self.calls += 1
            
            return score, False
        except Exception as e:
            print(f"Property calculation error: {e}")
            return 0.0, False

    def evaluate_with_components(self, smiles):
        """
        Evaluate molecule and return component decomposition values (for multi-objective optimization only)
        
        Returns:
            dict: {'logp': float, 'qed': float, 'combined': float} or {'combined': float}
        """
        if self.target_property == 'logp_qed':
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return {'logp': 0.0, 'qed': 0.0, 'combined': 0.0}
                
                # Get original LogP and QED values
                logp_value = float(self.logp_evaluator(smiles))
                qed_value = float(self.qed_evaluator(smiles))
                
                # Calculate normalized and combined values (same logic as in _fallback_evaluate)
                import math
                normalized_logp = 1 / (1 + math.exp(-0.5 * logp_value))
                normalized_qed = qed_value
                
                logp_weight = self.multi_objective_weights.get('logp', 0.5)
                qed_weight = self.multi_objective_weights.get('qed', 0.5)
                
                total_weight = logp_weight + qed_weight
                if total_weight > 0:
                    logp_weight /= total_weight
                    qed_weight /= total_weight
                
                combined_score = logp_weight * normalized_logp + qed_weight * normalized_qed
                
                return {
                    'logp': logp_value,
                    'qed': qed_value,
                    'combined': combined_score
                }
            except Exception as e:
                print(f"Warning: Failed to get components for {smiles}: {e}")
                return {'logp': 0.0, 'qed': 0.0, 'combined': 0.0}
        else:
            # For single-objective optimization, only return combined value
            combined_score = self.evaluate(smiles)
            return {'combined': combined_score}
    
    def reset_cache_for_experiment(self):
        """Reset cache for new script run initialization"""
        self.cache.clear()
        self.calls = 0           # Real call count
        self.total_calls = 0     # Total call count
        if not ORACLE_SILENT_MODE:
            print(f"Oracle cache reset, starting new script run (instance: {self.instance_id})")
    
    def evaluate_all_properties(self, smiles: str) -> dict:
        """
        Evaluate all supported properties of a molecule
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            dict: Dictionary containing all property values {'qed': float, 'logp': float, 'sa': float, ...}
        """
        if not smiles:
            return {prop: 0.0 for prop in ['qed', 'logp', 'sa', 'jnk3', 'gsk3b', 'drd2']}
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {prop: 0.0 for prop in ['qed', 'logp', 'sa', 'jnk3', 'gsk3b', 'drd2']}
        
        # Use evaluate_specific_properties to avoid code duplication and repeated evaluator creation
        all_properties = ['qed', 'logp', 'sa', 'jnk3', 'gsk3b', 'drd2']
        return self.evaluate_specific_properties(smiles, all_properties)
    
    def evaluate_specific_properties(self, smiles: str, properties: list) -> dict:
        """
        Evaluate only specified molecular properties (saves computational resources)
        
        Args:
            smiles: SMILES string of the molecule
            properties: List of properties to evaluate, e.g., ['qed', 'logp', 'sa']
            
        Returns:
            dict: Dictionary containing specified property values {property: value}
        """
        if not smiles:
            return {prop: 0.0 for prop in properties}
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {prop: 0.0 for prop in properties}
        
        results = {}
        
        for prop in properties:
            try:
                # Use global singleton evaluator to avoid repeated creation (reduces "Found local copy" messages)
                if prop not in self.property_evaluators:
                    property_to_oracle = {
                        'qed': 'QED',
                        'logp': 'LogP',
                        'sa': 'SA',
                        'jnk3': 'JNK3',
                        'gsk3b': 'GSK3B',
                        'drd2': 'DRD2'
                    }
                    
                    if prop in property_to_oracle:
                        oracle_name = property_to_oracle[prop]
                        self.property_evaluators[prop] = get_or_create_tdc_oracle(oracle_name)
                    else:
                        print(f"Warning: Unknown property {prop}, setting to 0.0")
                        results[prop] = 0.0
                        continue
                
                # Use cached evaluator for evaluation
                evaluator = self.property_evaluators[prop]
                results[prop] = float(evaluator(smiles))
                
            except Exception as e:
                print(f"{prop.upper()} evaluation failed: {e}")
                if prop == 'sa':
                    results[prop] = 5.0  # Default value, 5.0 indicates difficult synthesis
                else:
                    results[prop] = 0.0
        
        return results
    
    def get_cache_stats(self):
        """Get cache statistics"""
        cache_hits = self.total_calls - self.calls  # Cache hit count
        cache_hit_ratio = cache_hits / self.total_calls if self.total_calls > 0 else 0
        
        return {
            'instance_id': self.instance_id,
            'real_oracle_calls': self.calls,        # Real Oracle call count
            'total_calls_including_cache': self.total_calls,  # Total call count (including cache)
            'cache_hits': cache_hits,               # Cache hit count
            'cache_size': len(self.cache),          # Cache entry count
            'cache_hit_ratio': cache_hit_ratio      # Cache hit ratio
        }
    
    def _fallback_evaluate(self, smiles, mol):
        """Fallback evaluation method for custom property calculations"""
        if self.target_property == 'mr':
            # MR (Molecular Refractivity) calculation
            from rdkit.Chem import Descriptors
            try:
                mr_value = Descriptors.MolMR(mol)
                return mr_value
            except Exception as e:
                print(f"Warning: Failed to calculate MR for {smiles}: {e}")
                return 0.0
        elif self.target_property == 'logp_qed':
            # Multi-objective: LogP + QED combination
            try:
                logp_value = float(self.logp_evaluator(smiles))
                qed_value = float(self.qed_evaluator(smiles))
                
                # Normalize LogP to [0, 1] range for better combination
                # LogP typically ranges from -3 to +8, we'll use a sigmoid-like normalization
                # This maps extreme values to 0-1 range smoothly
                import math
                normalized_logp = 1 / (1 + math.exp(-0.5 * logp_value))  # Sigmoid normalization
                
                # QED is already in [0, 1] range
                normalized_qed = qed_value
                
                # Weighted combination using configurable weights
                logp_weight = self.multi_objective_weights.get('logp', 0.5)
                qed_weight = self.multi_objective_weights.get('qed', 0.5)
                
                # Ensure weights sum to 1
                total_weight = logp_weight + qed_weight
                if total_weight > 0:
                    logp_weight /= total_weight
                    qed_weight /= total_weight
                
                combined_score = logp_weight * normalized_logp + qed_weight * normalized_qed
                
                return combined_score
            except Exception as e:
                print(f"Warning: Failed to calculate LogP+QED for {smiles}: {e}")
                return 0.0
        else:
            return 0.0
    
    def save_cache(self, file_path):
        """Save cache to file"""
        with open(file_path, 'w') as f:
            yaml.dump(self.cache, f)
    
    def load_cache(self, file_path):
        """Load cache from file"""
        with open(file_path, 'r') as f:
            self.cache = yaml.safe_load(f) or {}
