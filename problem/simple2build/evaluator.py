import numpy as np
import random
from pathlib import Path
import pickle
from constellaration import problems
from constellaration.geometry import surface_rz_fourier
from constellaration import forward_model

from algorithm.base import Item
import datasets
from pprint import pprint
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
def convert2str(r_cos,z_sin):
    r_cos = np.array2string(r_cos, separator=', ', precision=8, suppress_small=True, max_line_width=1000)
    z_sin = np.array2string(z_sin, separator=', ', precision=8, suppress_small=True, max_line_width=1000)
    # 拼接为完整字符串
    final_string = (
    "r_cos = np.array(" + r_cos + ")\n\n"
    "z_sin = np.array(" + z_sin + ")"
    )
    return final_string

def evaluate_surface(r_cos,z_sin):
    surface = surface_rz_fourier.SurfaceRZFourier(r_cos=r_cos,z_sin=z_sin,
                                                  r_sin=None,   
                                                  z_cos=None,
                                                  n_field_periods=3,  
                                                  is_stellarator_symmetric=True)
    result,metrics = problems.SimpleToBuildQIStellarator().evaluate(surface,return_metrics=True)
    return result,metrics

def _evaluate_one_static(item):
    try:
        scope = {}
        exec(item.value, {"np": np}, scope)
        r_cos = scope['r_cos']
        z_sin = scope['z_sin']
        assert r_cos.shape == z_sin.shape 

        result, metrics = evaluate_surface(r_cos, z_sin)
        results_dict = {
            'original_results': {
                'l_delta_b': metrics.minimum_normalized_magnetic_gradient_scale_length,
                'feasibility':result.feasibility,
            },
            'transformed_results': {
                'l_delta_b': - result.score,
                'feasibility':result.feasibility,
            },
            'constraint_results': {
                'current_shape':r_cos.shape,
                'edge_rotational_transform_over_n_field_periods':metrics.edge_rotational_transform_over_n_field_periods,
                'qi':metrics.qi,
                'edge_magnetic_mirror_ratio':metrics.edge_magnetic_mirror_ratio,
                'aspect_ratio': metrics.aspect_ratio,
                'max_elongation':metrics.max_elongation,
                'feasibility':result.feasibility,
                'average_triangularity':metrics.average_triangularity,
                'vacuum_well':metrics.vacuum_well,
                'flux_compression_in_regions_of_bad_curvature':metrics.flux_compression_in_regions_of_bad_curvature
            },
            'overall_score': result.score
        }
        #print(results_dict)
        print(f'score: {result.score}, feasibility: {result.feasibility}')
        item.assign_results(results_dict)
        item.value = convert2str(r_cos, z_sin)
        return item

    except Exception as e:
        print(f"[Evaluation Error] Skipping item due to: {e}")
        import traceback
        traceback.print_exc()
        return None

class RewardingSystem:
    def __init__(self, config=None):
        self.config = config

    def evaluate(self, items):
        valid_items = []
        log_dict = {}
        finished = 0
        cpu_count = 10
        print(f'start evaluating')
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {executor.submit(_evaluate_one_static, item): item for item in items}

            for future in as_completed(futures):
                try:
                    result_item = future.result()  # 6min
                    finished += 1
                    print(f'Finished {finished}/{len(items)}')
                    if result_item is not None:
                        valid_items.append(result_item)
                except Exception as e:
                    print(f"[Future Error] Exception during evaluation: {e}")

        log_dict['invalid_num'] = len(items) - len(valid_items)
        log_dict['repeated_num'] = 0  # default is 0 unless you handle duplicates

        return valid_items, log_dict

def generate_initial_population(config,seed=42,n_sample=100):
    with open('/root/nian/MOLLM/problem/simple2build/init_items.pkl','rb') as f:
        items = pickle.load(f)
    reward_system = RewardingSystem(config)
    for i in range(len(items)):
        items[i].property_list = ['l_delta_b', 'feasibility']
    items, _ = reward_system.evaluate(items)
    return items


def get_database(config,seed=42,n_sample=100):
    '''
    ds = datasets.load_dataset(
    "proxima-fusion/constellaration",
    split="train",
    num_proc=4,
    )
    ds = ds.select_columns([c for c in ds.column_names
                            if c.startswith("boundary.")
                            or c.startswith("metrics.")])
    ds = ds.filter(
        lambda x: x == 3,
        input_columns=["boundary.n_field_periods"],
        num_proc=4,
    )
    ml_ds = ds.remove_columns([
        "boundary.n_field_periods", "boundary.is_stellarator_symmetric",  # all same value
        "boundary.r_sin", "boundary.z_cos",  # empty
        "boundary.json", "metrics.json", "metrics.id",  # not needed
    ])
    df = ml_ds.to_pandas()
    problem = problems.SimpleToBuildQIStellarator()
    feasibilities = []
    feasible_metrics = []
    violations = []
    for _,row in df.iterrows():
        metrics = forward_model.ConstellarationMetrics(
            aspect_ratio=row['metrics.aspect_ratio'],
            aspect_ratio_over_edge_rotational_transform=row['metrics.aspect_ratio_over_edge_rotational_transform'],
            axis_rotational_transform_over_n_field_periods=row['metrics.axis_rotational_transform_over_n_field_periods'],
            axis_magnetic_mirror_ratio=row['metrics.axis_magnetic_mirror_ratio'],
            average_triangularity=row['metrics.average_triangularity'],
            edge_rotational_transform_over_n_field_periods=row['metrics.edge_rotational_transform_over_n_field_periods'],
            qi=row['metrics.qi'],
            edge_magnetic_mirror_ratio=row['metrics.edge_magnetic_mirror_ratio'],
            max_elongation=row['metrics.max_elongation'],
            minimum_normalized_magnetic_gradient_scale_length=row['metrics.minimum_normalized_magnetic_gradient_scale_length'],
            flux_compression_in_regions_of_bad_curvature=row['metrics.flux_compression_in_regions_of_bad_curvature'],
            vacuum_well=row['metrics.vacuum_well'],
        )
        if problem.is_feasible(metrics):
            feasible_metrics.append(metrics)
        feasibilities.append(problem.compute_feasibility(metrics))
        violations.append(problem._normalized_constraint_violations(metrics))
    df['feasibility'] = feasibilities
    df = df.sort_values('feasibility',ascending=True).reset_index(drop=True)
    '''
    df = pd.read_csv('/root/nian/MOLLM/problem/simple2build/simple2build_databse.csv')
    df= df[:n_sample]
    items = []
    for _,row in df.iterrows():
        results_dict = {
            'original_results': {
                'l_delta_b': row['metrics.minimum_normalized_magnetic_gradient_scale_length'],
                'feasibility':row['feasibility'],
            },
            'transformed_results': {
                'l_delta_b': -row['metrics.minimum_normalized_magnetic_gradient_scale_length']/20,
                'feasibility':row['feasibility'],
            },
            'constraint_results': {
                'edge_rotational_transform_over_n_field_periods':row['metrics.edge_rotational_transform_over_n_field_periods'],
                'qi':row['metrics.qi'],
                'edge_magnetic_mirror_ratio':row['metrics.edge_magnetic_mirror_ratio'],
                'aspect_ratio': row['metrics.aspect_ratio'],
                'max_elongation':row['metrics.max_elongation'],
                'feasibility':row['feasibility'],
                'average_triangularity':row['metrics.average_triangularity'],
                'vacuum_well':row['metrics.vacuum_well'],
                'flux_compression_in_regions_of_bad_curvature':row['metrics.flux_compression_in_regions_of_bad_curvature']
            },
        }
        results_dict['overall_score'] = 0
        r_cos = np.stack(row['boundary.r_cos'])
        z_sin = np.stack(row['boundary.z_sin'])
        item = Item(convert2str(r_cos,z_sin),['l_delta_b','feasibility'])
        item.assign_results(results_dict)
        items.append(item)
    #with open('/home/hp/src/MOLLM/problem/fusion/best_items.pkl', 'rb') as f:
    #    loaded_items = pickle.load(f)
    #items.extend(loaded_items)
   
    return items
    