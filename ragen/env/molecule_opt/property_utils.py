"""
Molecular property description utilities
Reuses property mapping and description generation logic from SFT training to ensure RL and SFT prompts are completely consistent
"""

# Define property full name mapping (consistent with PROPERTY_FULL_NAMES in SFT)
PROPERTY_FULL_NAMES = {
    "qed": ["QED", "Quantitative Estimate of Drug-likeness (QED)", "drug-likeness quantified by QED score"],
    "logp": ["LogP", "Octanol-water partition coefficient (logP)", "lipophilicity measured by logP"],
    "jnk3": ["JNK3 inhibition", "c-Jun N-terminal kinase 3 inhibition probability", "inhibition probability of c-Jun N-terminal kinase 3"],
    "drd2": ["DRD2 inhibition", "Dopamine receptor D2 inhibition probability", "inhibition probability of Dopamine receptor D2"],
    "sa": ["SA score", "Synthetic accessibility score", "synthetic accessibility score (lower is better)"]
}

# Define optimization direction for each property
PROPERTY_OPTIMIZATION_DIRECTIONS = {
    "qed": "maximize",    # Higher QED is better (drug-likeness)
    "logp": "maximize",   # Usually want to increase LogP (lipophilicity)
    "jnk3": "maximize",   # Higher JNK3 inhibition activity is better
    "drd2": "maximize",   # Higher DRD2 inhibition activity is better
    "sa": "minimize"      # Lower SA score is better (easier to synthesize)
}

def generate_property_description(task: str, instr_setting: str = 'seen') -> str:
    """
    Generate property description based on task (logic completely consistent with SFT)
    
    Args:
        task: Task string, e.g. "qed+logp+sa"
        instr_setting: Instruction setting, seen or unseen (RL training defaults to seen)
    
    Returns:
        Property description string, e.g. "increase QED and LogP and decrease SA score"
    """
    properties = task.split('+')
    descriptions = []
    
    for prop in properties:
        if prop not in PROPERTY_FULL_NAMES:
            raise ValueError(f"Unknown property: {prop}. Supported properties: {list(PROPERTY_FULL_NAMES.keys())}")
        
        # Select property name based on seen/unseen
        if instr_setting == 'unseen':
            prop_name = PROPERTY_FULL_NAMES[prop][-1]  # Use unseen description
        else:
            prop_name = PROPERTY_FULL_NAMES[prop][0]   # Use seen description
        
        # Determine change direction (SA needs to decrease, others increase)
        change_direction = 'decrease' if prop == 'sa' else 'increase'
        descriptions.append(f"{change_direction} {prop_name}")
    
    # Connect with commas and "and"
    if len(descriptions) == 1:
        return descriptions[0]
    elif len(descriptions) == 2:
        return f"{descriptions[0]} and {descriptions[1]}"
    else:
        return ', '.join(descriptions[:-1]) + f" and {descriptions[-1]}"

def parse_task_config(env_config: dict) -> tuple[str, str]:
    """
    Parse task information in environment configuration, generate description and similarity threshold
    
    Args:
        env_config: Environment configuration dictionary, should contain molecule_opt_task and molecule_opt_similarity_threshold
    
    Returns:
        tuple: (task_description, similarity_threshold_str)
    """
    # Get task configuration
    task = env_config.get('molecule_opt_task', 'qed')
    similarity_threshold = env_config.get('molecule_opt_similarity_threshold', 0.4)
    
    # Generate task description
    task_description = generate_property_description(task, 'seen')
    
    # Format similarity threshold
    similarity_threshold_str = str(similarity_threshold)
    
    return task_description, similarity_threshold_str

def get_supported_properties() -> list[str]:
    """Return list of all supported properties"""
    return list(PROPERTY_FULL_NAMES.keys())

def get_optimization_directions(task: str) -> dict[str, str]:
    """
    Get optimization direction for each property in the task
    
    Args:
        task: Task string, e.g. "qed+logp+sa"
    
    Returns:
        dict: Optimization direction for each property {"qed": "maximize", "sa": "minimize", ...}
    """
    properties = task.split('+')
    directions = {}
    for prop in properties:
        if prop in PROPERTY_OPTIMIZATION_DIRECTIONS:
            directions[prop] = PROPERTY_OPTIMIZATION_DIRECTIONS[prop]
        else:
            # Unknown properties default to maximize
            directions[prop] = "maximize"
    return directions

def calculate_success_rates(task: str, best_results_dict: dict, improvement_based: bool = True) -> dict:
    """
    Calculate task success rate metrics, supports two modes: improvement threshold and absolute threshold
    
    Args:
        task: Task string, e.g. "qed" or "qed+logp+sa"
        best_results_dict: Global best results dictionary {smiles: {best_score, initial_score, ...}}
        improvement_based: Whether to use improvement-based success rate (True) vs absolute threshold success rate (False)
    
    Returns:
        dict: Success rate metrics, containing results from both modes
    """
    if not best_results_dict:
        return {}
    
    properties = task.split('+')
    results = {}
    total_molecules = len(best_results_dict)
    
    # Initialize counters
    improvement_success_counts = {prop: 0 for prop in properties}
    absolute_success_counts = {prop: 0 for prop in properties}
    improvement_overall_success = 0
    absolute_overall_success = 0
    
    for smiles, data in best_results_dict.items():
        if 'best_score' not in data or 'initial_score' not in data:
            continue
        
        improvement_all_success = True
        absolute_all_success = True
        
        # Handle multi-objective tasks
        if len(properties) > 1:
            individual_improvements = data.get('individual_improvements', {})
            individual_new_properties = data.get('individual_new_properties', {})
            
            for prop in properties:
                # 1. Improvement threshold calculation
                if individual_improvements:
                    improvement = individual_improvements.get(prop, 0)
                    if isinstance(improvement, dict):
                        improvement = improvement.get('improvement', 0)
                    
                    if is_property_success(prop, improvement):
                        improvement_success_counts[prop] += 1
                    else:
                        improvement_all_success = False
                else:
                    improvement_all_success = False
                
                # 2. Absolute threshold calculation
                if individual_new_properties:
                    absolute_value = individual_new_properties.get(prop)
                    if absolute_value is not None:
                        if is_property_absolute_success(prop, absolute_value):
                            absolute_success_counts[prop] += 1
                        else:
                            absolute_all_success = False
                    else:
                        absolute_all_success = False
                else:
                    absolute_all_success = False
        
        # Handle single-objective tasks
        else:
            prop = properties[0]
            best_score = data['best_score']
            initial_score = data['initial_score']
            improvement = best_score - initial_score
            
            # 1. Improvement threshold calculation
            if is_property_success(prop, improvement):
                improvement_success_counts[prop] += 1
            else:
                improvement_all_success = False
            
            # 2. Absolute threshold calculation
            if is_property_absolute_success(prop, best_score):
                absolute_success_counts[prop] += 1
            else:
                absolute_all_success = False
        
        if improvement_all_success:
            improvement_overall_success += 1
        
        if absolute_all_success:
            absolute_overall_success += 1
    
    # Return both types of success rate results
    if improvement_based:
        # Improvement threshold success rate
        for prop in properties:
            results[f"{prop}_success_rate"] = improvement_success_counts[prop] / total_molecules
        
        if len(properties) > 1:
            results["overall_success_rate"] = improvement_overall_success / total_molecules
    else:
        # Absolute threshold success rate
        for prop in properties:
            results[f"{prop}_absolute_success_rate"] = absolute_success_counts[prop] / total_molecules
        
        if len(properties) > 1:
            results["overall_absolute_success_rate"] = absolute_overall_success / total_molecules
    
    return results

def calculate_both_success_rates(task: str, best_results_dict: dict) -> dict:
    """
    Calculate both improvement threshold and absolute threshold success rates simultaneously
    
    Args:
        task: Task string
        best_results_dict: Global best results dictionary
    
    Returns:
        dict: Complete results containing both types of success rates
    """
    if not best_results_dict:
        return {}
    
    # Calculate improvement threshold success rate
    improvement_results = calculate_success_rates(task, best_results_dict, improvement_based=True)
    
    # Calculate absolute threshold success rate  
    absolute_results = calculate_success_rates(task, best_results_dict, improvement_based=False)
    
    # Merge results
    results = {}
    results.update(improvement_results)
    results.update(absolute_results)
    
    return results

def calculate_property_averages(task: str, best_results_dict: dict) -> dict:
    """
    Calculate average metrics for each property (avg_improvement, avg_best_score)
    
    Args:
        task: Task string, e.g. "qed" or "qed+logp+sa"
        best_results_dict: Global best results dictionary
    
    Returns:
        dict: Average metrics {"{prop}_avg_improvement": value, "{prop}_avg_best_score": value}
    """
    if not best_results_dict:
        return {}
    
    properties = task.split('+')
    results = {}
    
    if len(properties) == 1:
        # Single-objective task: direct calculation
        prop = properties[0]
        improvements = []
        best_scores = []
        
        for data in best_results_dict.values():
            if 'best_score' in data and 'initial_score' in data:
                improvement = data['best_score'] - data['initial_score']
                improvements.append(improvement)
                best_scores.append(data['best_score'])
        
        if improvements:
            results[f"{prop}_avg_improvement"] = sum(improvements) / len(improvements)
            results[f"{prop}_avg_best_score"] = sum(best_scores) / len(best_scores)
            
    else:
        # Multi-objective task: use individual data
        property_improvements = {prop: [] for prop in properties}
        property_best_scores = {prop: [] for prop in properties}
        
        for data in best_results_dict.values():
            individual_improvements = data.get('individual_improvements', {})
            individual_old_properties = data.get('individual_old_properties', {})
            individual_new_properties = data.get('individual_new_properties', {})
            
            for prop in properties:
                # Prioritize using actual best values from individual_new_properties
                if prop in individual_new_properties:
                    property_best_scores[prop].append(individual_new_properties[prop])
                    
                    # Calculate improvement
                    if prop in individual_old_properties:
                        improvement = individual_new_properties[prop] - individual_old_properties[prop]
                        property_improvements[prop].append(improvement)
                    elif prop in individual_improvements:
                        # If no old_properties, use stored improvement
                        property_improvements[prop].append(individual_improvements[prop])
                
                # If no new_properties, try to reconstruct from improvement and old_properties
                elif prop in individual_improvements and prop in individual_old_properties:
                    improvement = individual_improvements[prop]
                    property_improvements[prop].append(improvement)
                    
                    # Calculate best value from initial + improvement
                    best_value = individual_old_properties[prop] + improvement
                    property_best_scores[prop].append(best_value)
                
                # Final fallback: only improvement available
                elif prop in individual_improvements:
                    prop_data = individual_improvements[prop]
                    if isinstance(prop_data, dict):
                        # Use improvement field
                        if 'improvement' in prop_data:
                            property_improvements[prop].append(prop_data['improvement'])
                        # Use best field
                        if 'best' in prop_data:
                            property_best_scores[prop].append(prop_data['best'])
                    else:
                        # If it's a direct numerical value, it represents improvement
                        property_improvements[prop].append(prop_data)
        
        # Calculate average values for each property
        for prop in properties:
            if property_improvements[prop]:
                avg_improvement = sum(property_improvements[prop]) / len(property_improvements[prop])
                results[f"{prop}_avg_improvement"] = avg_improvement
            
            if property_best_scores[prop]:
                avg_best_score = sum(property_best_scores[prop]) / len(property_best_scores[prop])
                results[f"{prop}_avg_best_score"] = avg_best_score
    
    return results

def is_property_success(property_name: str, improvement: float) -> bool:
    """
    Determine if a single property is successful based on improvement
    
    Args:
        property_name: Property name
        improvement: Improvement amount (best_score - initial_score)
    
    Returns:
        bool: Whether success threshold is reached
    """
    config = PROPERTY_SUCCESS_THRESHOLDS.get(property_name)
    if not config:
        return False
        
    threshold = config["threshold"]
    direction = config["direction"]
    
    if direction == "increase":
        return improvement >= threshold
    else:  # direction == "decrease" (SA)
        return improvement <= -threshold  # Negative improvement indicates decrease

def is_property_absolute_success(property_name: str, absolute_value: float) -> bool:
    """
    Determine if a single property reaches absolute threshold based on absolute value
    
    Args:
        property_name: Property name
        absolute_value: Absolute score value of the molecule for this property
    
    Returns:
        bool: Whether absolute threshold standard is met
    """
    config = PROPERTY_ABSOLUTE_THRESHOLDS.get(property_name)
    if not config:
        return False
        
    threshold = config["threshold"]
    direction = config["direction"]
    
    if direction == "increase":
        return absolute_value >= threshold
    else:  # direction == "decrease" (SA)
        return absolute_value <= threshold  # Lower SA is better

# Weight configuration based on improvement difficulty
PROPERTY_DIFFICULTY_WEIGHTS = {
    "jnk3": 12,     # Hardest: typical improvement 0.05-0.1
    "qed": 10,       # Medium: typical improvement 0.1-0.2
    "drd2": 4,      # Medium-easy: typical improvement 0.25
    "sa": 2,        # Easier: typical improvement 0.5
    "logp": 1       # Easiest: typical improvement 2-3
}

# Success threshold configuration (minimum values representing significant improvement)
PROPERTY_SUCCESS_THRESHOLDS = {
    "qed": {"threshold": 0.1, "direction": "increase"},
    "logp": {"threshold": 1.0, "direction": "increase"},
    "jnk3": {"threshold": 0.1, "direction": "increase"},
    "drd2": {"threshold": 0.5, "direction": "increase"},
    "sa": {"threshold": 0.5, "direction": "decrease"}
}

# Absolute value threshold configuration (for judging if molecular properties meet absolute standards)
PROPERTY_ABSOLUTE_THRESHOLDS = {
    "qed": {"threshold": 0.9, "direction": "increase"},     # QED ≥ 0.9
    "logp": {"threshold": 2.0, "direction": "increase"},   # LogP ≥ 2.0
    "drd2": {"threshold": 0.8, "direction": "increase"},   # DRD2 ≥ 0.8
    "sa": {"threshold": 2.5, "direction": "decrease"},     # SA ≤ 2.5 (lower is better)
    "gsk3b": {"threshold": 0.4, "direction": "increase"}   # GSK3B ≥ 0.4
}

def calculate_multi_objective_reward(improvements: dict, task_properties: list, optimization_directions: dict = None) -> tuple[float, dict]:
    """
    Calculate reward for multi-objective tasks
    
    Args:
        improvements: Dictionary of improvement values for each property {property: improvement_value}
        task_properties: List of properties involved in current task
        optimization_directions: Dictionary of optimization directions for each property {property: "maximize"/"minimize"}
    
    Returns:
        tuple: (total_reward, reward_info)
    """
    # If no optimization directions provided, use defaults
    if optimization_directions is None:
        optimization_directions = {prop: PROPERTY_OPTIMIZATION_DIRECTIONS.get(prop, "maximize") 
                                 for prop in task_properties}
    
    # 1. Base reward: weighted improvements
    base_reward = 0.0
    weighted_improvements = {}
    
    for prop in task_properties:
        improvement = improvements.get(prop, 0.0)
        weight = PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0)
        direction = optimization_directions.get(prop, "maximize")
        
        if direction == "minimize":  # minimize: negative improvement indicates decrease (good)
            weighted_improvement = (-improvement) * weight
        else:  # maximize: positive improvement indicates increase (good)
            weighted_improvement = improvement * weight
            
        weighted_improvements[prop] = weighted_improvement
        base_reward += weighted_improvement
    
    # 2. Success bonus: check if all properties reach success threshold
    success_bonus = 0.0
    success_status = {}
    all_success = True
    
    for prop in task_properties:
        improvement = improvements.get(prop, 0.0)
        config = PROPERTY_SUCCESS_THRESHOLDS.get(prop)
        
        if config:
            threshold = config["threshold"]
            direction = config["direction"]
            
            if direction == "increase":
                success = improvement >= threshold
            else:  # direction == "decrease" (SA)
                success = improvement <= -threshold  # Negative improvement indicates decrease
            
            success_status[prop] = {
                "success": success,
                "improvement": improvement,
                "threshold": threshold,
                "direction": direction
            }
            
            if not success:
                all_success = False
        else:
            # If no threshold configured, consider as not participating in success evaluation
            success_status[prop] = {"success": True, "improvement": improvement}
    
    # If all properties succeed and it's multi-objective task, give huge bonus
    if all_success and len(task_properties) > 1:
        # Adjust success bonus based on task complexity
        success_bonus = 5.0 + len(task_properties) * 2.0
    
    total_reward = base_reward + success_bonus
    
    reward_info = {
        "base_reward": base_reward,
        "success_bonus": success_bonus,
        "total_reward": total_reward,
        "weighted_improvements": weighted_improvements,
        "success_status": success_status,
        "all_success": all_success
    }
    
    return total_reward, reward_info

# Property-specific improvement suggestions
PROPERTY_IMPROVEMENT_GUIDANCE = {
    "qed": {
        "positive": "Great QED improvement! The molecule is becoming more drug-like.",
        "negative": "QED decreased. Try to improve QED.",
        "maintain": "Excellent drug-likeness maintained."
    },
    "logp": {
        "positive": "Good LogP increase! Enhanced lipophilicity will improve membrane permeability.", 
        "negative": "LogP decreased. Try to improve LogP.",
        "maintain": "Good lipophilicity balance maintained."
    },
    "sa": {
        "positive": "Excellent! SA score decreased, making the molecule easier to synthesize.",
        "negative": "SA score increased, making synthesis more challenging. Try to decrease SA score.",
        "maintain": "Good synthetic accessibility maintained."
    },
    "jnk3": {
        "positive": "Great JNK3 activity improvement! Enhanced kinase inhibition potential.",
        "negative": "JNK3 activity decreased. Try to improve JNK3 inhibition.",
        "maintain": "Strong JNK3 activity maintained."
    },
    "drd2": {
        "positive": "Great DRD2 activity improvement! Better dopamine receptor interaction.",
        "negative": "DRD2 activity decreased. Try to improve DRD2 inhibition.",
        "maintain": "Strong DRD2 activity maintained."
    }
}

def generate_multi_objective_guidance(reward_info: dict) -> str:
    """
    Generate guidance information for multi-objective tasks
    
    Args:
        reward_info: Reward information dictionary containing improvements, success_status, etc.
        
    Returns:
        str: Guidance information string
    """
    if not reward_info.get("is_multi_objective", False):
        return ""
    
    improvements = reward_info.get("improvements", {})
    success_status = reward_info.get("success_status", {})
    all_success = reward_info.get("all_success", False)
    success_bonus = reward_info.get("success_bonus", 0)
    
    # Analyze each property situation
    positive_changes = []  # Positive improvements
    negative_changes = []  # Negative changes
    achieved_thresholds = []  # Properties that achieved thresholds
    
    for prop, improvement in improvements.items():
        status = success_status.get(prop, {})
        is_success = status.get("success", False)
        
        if improvement > 0:
            # Positive improvement
            if prop == 'sa':
                # SA special handling: positive improvement means harder to synthesize (bad)
                negative_changes.append({
                    "prop": prop,
                    "improvement": improvement,
                    "message": PROPERTY_IMPROVEMENT_GUIDANCE[prop]["negative"]
                })
            else:
                # Other properties: positive improvement is good
                message = PROPERTY_IMPROVEMENT_GUIDANCE[prop]["positive"]
                if is_success:
                    message += f" 🎯 Threshold achieved (+{improvement:.3f} ≥ {status.get('threshold', 0)})!"
                
                positive_changes.append({
                    "prop": prop,
                    "improvement": improvement,
                    "message": message,
                    "threshold_achieved": is_success
                })
                
        elif improvement < 0:
            # Negative change
            if prop == 'sa':
                # SA special handling: negative improvement means easier to synthesize (good)
                message = PROPERTY_IMPROVEMENT_GUIDANCE[prop]["positive"]
                if is_success:
                    message += f" 🎯 Threshold achieved ({improvement:.3f} ≤ -{status.get('threshold', 0)})!"
                
                positive_changes.append({
                    "prop": prop,
                    "improvement": improvement,
                    "message": message,
                    "threshold_achieved": is_success
                })
            else:
                # Other properties: negative improvement is bad
                negative_changes.append({
                    "prop": prop,
                    "improvement": improvement,
                    "message": PROPERTY_IMPROVEMENT_GUIDANCE[prop]["negative"]
                })
    
    # Generate guidance information
    guidance_parts = []
    
    # 1. Success bonus celebration (if any)
    if all_success and success_bonus > 0:
        guidance_parts.append(f"🎉 OUTSTANDING! All targets achieved! Success bonus: +{success_bonus:.1f}")
    
    # 2. Positive improvement feedback
    if positive_changes:
        if len(positive_changes) == 1:
            guidance_parts.append(positive_changes[0]["message"])
        else:
            messages = [change["message"] for change in positive_changes]
            guidance_parts.append(" ".join(messages))
    
    # 3. Negative change feedback (if any)
    if negative_changes:
        if positive_changes:
            guidance_parts.append("However, some properties need attention:")
        for change in negative_changes:
            guidance_parts.append(change["message"])
    
    # 4. Overall suggestions
    if all_success:
        guidance_parts.append("Continue with similar modification strategies!")
    elif positive_changes and not negative_changes:
        guidance_parts.append("Excellent progress! Keep refining with this approach.")
    elif positive_changes and negative_changes:
        guidance_parts.append("Focus on maintaining the positive changes while addressing the declining properties.")
    else:
        guidance_parts.append("Consider trying different structural modifications.")
    
    return " ".join(guidance_parts)

def format_property_scores(original_scores: dict, new_scores: dict, task_properties: list) -> str:
    """
    Format property score display for original and new molecules
    
    Args:
        original_scores: Property scores of original molecule {property: score}
        new_scores: Property scores of new molecule {property: score}
        task_properties: List of properties involved in current task
        
    Returns:
        str: Formatted score display string
    """
    score_parts = []
    
    # Display each property: original score -> new score (change)
    for prop in task_properties:
        original_score = original_scores.get(prop, 0.0)
        new_score = new_scores.get(prop, 0.0)
        improvement = new_score - original_score
        
        # Determine change direction indicator
        if improvement > 0:
            if prop == 'sa':
                change_symbol = "↑"  # SA increase is bad
            else:
                change_symbol = "↑"  # Other properties increase is good
        elif improvement < 0:
            if prop == 'sa':
                change_symbol = "↓"  # SA decrease is good
            else:
                change_symbol = "↓"  # Other properties decrease is bad
        else:
            change_symbol = "→"  # No change
        
        # Get property display name
        prop_name = PROPERTY_FULL_NAMES[prop][0]  # Use short name
        
        score_parts.append(f"{prop_name}: {original_score:.3f} → {new_score:.3f} ({change_symbol}{abs(improvement):.3f})")
    
    return " | ".join(score_parts)

def generate_complete_multi_objective_guidance(
    original_scores: dict, 
    new_scores: dict, 
    task_properties: list,
    reward_info: dict,
    is_initial: bool = False
) -> str:
    """
    Generate complete multi-objective guidance including score display and advice
    
    Args:
        original_scores: Property scores of original molecule
        new_scores: Property scores of new molecule  
        task_properties: Task property list
        reward_info: Reward information
        is_initial: Whether this is initial display (showing original molecule info)
        
    Returns:
        str: Complete guidance string
    """
    guidance_parts = []
    
    # 1. If initial state, display original molecule information
    if is_initial:
        # For initial state, get original molecule property scores from new_scores
        original_score_str = ", ".join([
            f"{PROPERTY_FULL_NAMES[prop][0]}: {new_scores.get(prop, 0.0):.3f}" 
            for prop in task_properties
        ])
        guidance_parts.append(f"Original molecule properties: {original_score_str}")
        return " ".join(guidance_parts)
    
    # 2. If global improvement info exists, show improvement relative to original molecule
    if 'global_improvements' in reward_info and 'original_properties' in reward_info:
        global_improvements = reward_info['global_improvements']
        original_properties = reward_info['original_properties']
        
        global_parts = []
        for prop in task_properties:
            if prop in global_improvements and prop in original_properties:
                global_imp = global_improvements[prop]
                orig_val = original_properties[prop]
                current_val = new_scores.get(prop, 0.0)
                
                # Determine if it's a good improvement
                if prop == 'sa':
                    symbol = "↓" if global_imp < 0 else "↑"
                    is_good = global_imp < 0
                else:
                    symbol = "↑" if global_imp > 0 else "↓"
                    is_good = global_imp > 0
                
                color_indicator = "✓" if is_good else "✗"
                global_parts.append(f"{PROPERTY_FULL_NAMES[prop][0]}: {orig_val:.3f}→{current_val:.3f} ({symbol}{abs(global_imp):.3f}{color_indicator})")
        
        if global_parts:
            guidance_parts.append(f"Properties (vs original): {' | '.join(global_parts)}")
    else:
        # If no global info, show step-by-step changes
        score_display = format_property_scores(original_scores, new_scores, task_properties)
        guidance_parts.append(f"Property changes: {score_display}")
    
    # 3. Add multi-objective guidance advice
    if len(task_properties) > 1:
        reward_info['is_multi_objective'] = True
        # Use global improvements
        if 'global_improvements' in reward_info:
            improvements = reward_info['global_improvements']
        else:
            improvements = {prop: new_scores.get(prop, 0.0) - original_scores.get(prop, 0.0) 
                           for prop in task_properties}
        reward_info['improvements'] = improvements
        
        # If weighted_score exists, display it
        if 'weighted_score' in reward_info:
            guidance_parts.append(f"Weighted score: {reward_info['weighted_score']:.3f}")
        
        guidance_advice = generate_multi_objective_guidance(reward_info)
        if guidance_advice:
            guidance_parts.append(guidance_advice)
    
    return " ".join(guidance_parts)

def validate_task(task: str) -> bool:
    """Validate if task string is valid"""
    try:
        properties = task.split('+')
        return all(prop in PROPERTY_FULL_NAMES for prop in properties)
    except:
        return False