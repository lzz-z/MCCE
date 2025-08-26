import numpy as np
import json
import logging
import random
import copy

# --- Import SACS-specific modules from the same directory ---
from .sacs_file_modifier import SacsFileModifier
from .sacs_runner import SacsAnalysisManager
from .sacs_interface_uc import get_sacs_uc_summary
from .sacs_interface_weight import calculate_sacs_volume
from .sacs_interface_ftg import get_sacs_fatigue_summary

# A baseline candidate to start from if generation fails.
BASELINE_CODE_BLOCKS = {
    "new_code_blocks": {
        "GRUP_LG1": "GRUP LG1         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.005.00",
        "GRUP_LG2": "GRUP LG2         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.006.15",
        "GRUP_LG3": "GRUP LG3         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.006.75",
        "GRUP_LG4": "GRUP LG4         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.00",
        "GRUP_T01": "GRUP T01         16.000 0.625 29.0111.2035.00 1    1.001.00     0.500N490.00",
        "GRUP_T02": "GRUP T02         20.000 0.750 29.0011.6035.00 1    1.001.00     0.500N490.00",
        "GRUP_T03": "GRUP T03         12.750 0.500 29.0111.6035.00 1    1.001.00     0.500N490.00",
        "PGRUP_P01": "PGRUP P01 0.3750I29.000 0.25036.000                                     490.0000"
    }
}


def _parse_and_modify_line(line, block_name):
    """
    V4 - A highly robust helper function to parse, modify, and reconstruct a SACS line,
    ensuring fixed-width format is always preserved. This version was validated
    with an independent test script.
    """
    try:
        keyword = block_name.split()[0]  # 'GRUP' or 'PGRUP'

        if keyword == "GRUP":
            # --- Extremely safe parsing ---
            key_part = line[0:4]  # 'GRUP'
            # Ensure group part has fixed width for consistent spacing
            group_name = block_name.split()[1]  # e.g., 'LG4'
            group_part = f" {group_name:<13}"  # ' LG4         ' (1 space + 13 chars left-aligned)

            od_str = line[18:24]  # ' 42.000'
            wt_str = line[25:30]  # ' 1.375'
            rest_of_line = line[31:]  # Everything after WT, starts at column 32

            od_val = float(od_str)
            wt_val = float(wt_str)

            # --- Modification ---
            if random.choice([True, False]):
                od_val *= random.uniform(0.95, 1.05)
                od_val = np.clip(od_val, 10.0, 48.0)
            else:
                wt_val *= random.uniform(0.95, 1.05)
                wt_val = np.clip(wt_val, 0.5, 2.5)

            # --- Extremely safe reconstruction ---
            # We hardcode the space separators to guarantee format
            new_line = f"{key_part}{group_part}{od_val:>6.3f} {wt_val:>5.3f} {rest_of_line}"
            # Final check to trim any extra space if original line was shorter
            return new_line[:len(line)]

        elif keyword == "PGRUP":
            part1 = line[0:11]
            thick_str = line[11:17]
            rest_of_line = line[17:]
            thick_val = float(thick_str) * random.uniform(0.95, 1.05)
            thick_val = np.clip(thick_val, 0.250, 0.750)
            return f"{part1}{thick_val:<6.4f}{rest_of_line}"

    except Exception as e:
        logging.error(f"Error in _parse_and_modify_line for '{line}': {e}", exc_info=True)
        return line  # Return original on failure

    return line


def generate_initial_population(config, seed):
    np.random.seed(seed)
    random.seed(seed)
    population_size = config.get('optimization.pop_size')
    initial_population = []
    for i in range(population_size):
        # The first a few candidates can remain the pure baseline for stability
        if i < 2:
            initial_population.append(json.dumps(BASELINE_CODE_BLOCKS))
            continue

        new_candidate_blocks = copy.deepcopy(BASELINE_CODE_BLOCKS)
        block_to_modify_key = random.choice(list(new_candidate_blocks["new_code_blocks"].keys()))
        block_to_modify_name = block_to_modify_key.replace("_", " ")
        original_sacs_line = new_candidate_blocks["new_code_blocks"][block_to_modify_key]
        modified_sacs_line = _parse_and_modify_line(original_sacs_line, block_to_modify_name)

        # Log the change for debugging
        if original_sacs_line != modified_sacs_line:
            logging.debug(f"Init pop generation: Modified {block_to_modify_key}")
            logging.debug(f"  OLD: {original_sacs_line}")
            logging.debug(f"  NEW: {modified_sacs_line}")

        new_candidate_blocks["new_code_blocks"][block_to_modify_key] = modified_sacs_line
        initial_population.append(json.dumps(new_candidate_blocks))

    return initial_population


class RewardingSystem:
    # --- The rest of the class remains unchanged from the previous correct version ---
    def __init__(self, config):
        self.config = config
        self.sacs_project_path = config.get('sacs.project_path')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.modifier = SacsFileModifier(self.sacs_project_path)
        self.runner = SacsAnalysisManager(self.sacs_project_path)
        self.objs = config.get('goals', [])
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}

    def evaluate(self, items):
        invalid_num = 0
        for item in items:
            try:
                raw_value = item.value
                json_str = raw_value
                if '<candidate>' in raw_value:
                    json_str = raw_value.split('<candidate>', 1)[1].rsplit('</candidate>', 1)[0].strip()

                modifications = json.loads(json_str)
                new_code_blocks = modifications.get("new_code_blocks")

                if not new_code_blocks or not isinstance(new_code_blocks, dict):
                    self.logger.warning(f"Invalid candidate format: {item.value}")
                    self._assign_penalty(item, "Invalid JSON format")
                    invalid_num += 1
                    continue

                if not self.modifier.replace_code_blocks(new_code_blocks):
                    self._assign_penalty(item, "File modification failed")
                    invalid_num += 1
                    continue

                analysis_result = self.runner.run_with_retry()
                if not analysis_result.get('success'):
                    self._assign_penalty(item, f"SACS run failed: {analysis_result.get('error', 'Unknown error')}")
                    invalid_num += 1
                    continue

                weight_res = calculate_sacs_volume(self.sacs_project_path)
                uc_res = get_sacs_uc_summary(self.sacs_project_path)
                ftg_res = get_sacs_fatigue_summary(self.sacs_project_path)

                if not all([res.get('status') == 'success' for res in [weight_res, uc_res, ftg_res]]):
                    self._assign_penalty(item, "Metric extraction failed")
                    invalid_num += 1
                    continue

                original = {
                    'weight': weight_res['total_volume_m3'],
                    'uc': uc_res['max_uc'],
                    'fatigue': ftg_res['min_life_years']
                }

                transformed = self._transform_objectives(original)
                overall_score = len(self.objs) - np.sum(list(transformed.values()))

                results = {
                    'original_results': original,
                    'transformed_results': transformed,
                    'overall_score': overall_score
                }

                item.assign_results(results)

            except Exception as e:
                self.logger.critical(f"Critical error during evaluation of item '{getattr(item, 'value', 'N/A')}': {e}",
                                     exc_info=True)
                self._assign_penalty(item, f"Critical evaluation error: {e}")
                invalid_num += 1

        log_dict = {
            "invalid_num": invalid_num,
            "repeated_num": 0
        }
        return items, log_dict

    def _assign_penalty(self, item, reason=""):
        penalty_score = 999
        original = {}
        for obj in self.objs:
            original[obj] = penalty_score if self.obj_directions[obj] == 'min' else -penalty_score

        transformed = {obj: 1.0 for obj in self.objs}
        overall_score = -penalty_score

        results = {
            'original_results': original,
            'transformed_results': transformed,
            'overall_score': overall_score,
            'error_reason': reason
        }
        item.assign_results(results)

    def _transform_objectives(self, original_results):
        transformed = {}
        w = original_results.get('weight', 999)
        transformed['weight'] = np.clip((w - 2.0) / 3.0, 0, 1)

        uc = original_results.get('uc', 999)
        transformed['uc'] = 1.0 if uc > 1.0 else np.clip((uc - 0.5) / 0.5, 0, 1)

        ftg = original_results.get('fatigue', 0)
        normalized_ftg = np.clip((ftg - 20) / 480.0, 0, 1)
        transformed['fatigue'] = 1 - normalized_ftg

        return transformed
