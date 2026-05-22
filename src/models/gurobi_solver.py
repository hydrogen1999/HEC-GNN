import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.sparse as sp
import dimod
from dimod import BinaryQuadraticModel

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class GurobiSolver:

    def __init__(self, instance_name: str, dataset: str, random_seed: int = None,
                 time_limit: int = 3600, threads: int = 8, output_flag: int = 0,
                 use_local_search: bool = False, run_name: str = None, **kwargs):
    
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is not available. Please install gurobipy.")

        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.time_limit = time_limit
        self.threads = threads
        self.output_flag = output_flag
        self.use_local_search = use_local_search
        
        self.solver_name = run_name if run_name else "gurobi"

        self.best_solution_time = None
        self.start_time = 0
        self.best_obj = float('inf')

        print(f"Initialized {self.solver_name} solver with time_limit={time_limit}s, threads={threads}. Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  time_limit: {self.time_limit}")
        print(f"  threads: {self.threads}")
        print(f"  output_flag: {self.output_flag}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def _mycallback(self, model: gp.Model, where: int) -> None:
        if where == gp.GRB.Callback.MIP:
            current_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            if current_obj < self.best_obj:
                self.best_obj = current_obj
                self.best_solution_time = time.time() - self.start_time

    def _create_gurobi_model(self, bqm: BinaryQuadraticModel) -> tuple:
        """Build the Gurobi model from the BQM."""
        model = gp.Model("Ising_QUBO")

        # Set parameters
        model.setParam("TimeLimit", self.time_limit)
        model.setParam("Threads", self.threads)
        model.setParam("OutputFlag", self.output_flag)
        model.setParam("MIPFocus", 1)  # Focus on finding good solutions quickly
        
        # Set random seed if provided
        if self.random_seed is not None:
            model.setParam("Seed", self.random_seed)

        # Variables
        x = {var: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{var}") for var in bqm.variables}

        # Linear terms
        linear_expr = gp.LinExpr(sum(bqm.linear[var] * x[var] for var in bqm.variables))

        # Quadratic terms
        quadratic_expr = gp.QuadExpr(sum(coeff * x[i] * x[j] for (i, j), coeff in bqm.quadratic.items()))

        # Objective: minimize linear + quadratic terms
        model.setObjective(linear_expr + quadratic_expr, gp.GRB.MINIMIZE)
        
        return model, x

    def solve(self, J: np.ndarray):
        n = J.shape[0]
        
        linear = {}
        quadratic = {}
        
        if sp.issparse(J):
            J = J.toarray()
        
        offset = -0.5 * np.sum(J)
        
        for i in range(n):
            linear[i] = 2.0 * (np.sum(J[i, :]) - J[i, i])
            
            quadratic[(i, i)] = -2.0 * J[i, i]
            
            for j in range(i + 1, n):
                quadratic[(i, j)] = -4.0 * J[i, j]

        bqm = BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)

        model, x = self._create_gurobi_model(bqm)

        self.start_time = time.time()
        self.best_obj = float("inf")
        self.best_solution_time = None

        try:
            model.optimize(self._mycallback)
            time_taken = time.time() - self.start_time

            if model.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT]:
                binary_solution = {var: int(round(x[var].X)) for var in bqm.variables}
                spins = np.array([2 * binary_solution[i] - 1 for i in range(n)], dtype=np.int8)

                obj_value = model.objVal + bqm.offset
                
                computed_energy = self.compute_energy(J, spins)

                gap = 0.0
                lower_bound = obj_value
                if model.status == gp.GRB.TIME_LIMIT:
                    lower_bound = model.objBound + bqm.offset
                    gap = abs(obj_value - lower_bound) / abs(obj_value) if abs(obj_value) > 1e-12 else 0.0

                self._store_results(
                    gurobi_energy=obj_value,
                    computed_energy=computed_energy,
                    spins=spins,
                    time_taken=time_taken,
                    gap=gap,
                    lower_bound=lower_bound,
                    status=model.status,
                    best_solution_time=self.best_solution_time
                )

            else:
                print(f"Gurobi optimization failed with status: {model.status}")
                return

        except Exception as e:
            print(f"Gurobi solver failed: {e}")
            return

    def compute_energy(self, J: np.ndarray, spins: np.ndarray) -> float:
        if sp.issparse(J):
            return -0.5 * spins.T @ J @ spins
        else:
            return -0.5 * np.dot(spins, np.dot(J, spins))

    def _store_results(self, gurobi_energy: float, computed_energy: float, spins: np.ndarray,
                      time_taken: float, gap: float, lower_bound: float, status: int,
                      best_solution_time: float):
        results_dir = Path(f"results/{self.dataset}/{self.solver_name}")
        results_dir.mkdir(parents=True, exist_ok=True)

        instance_base = self.instance_name.replace('.txt', '')
        csv_file = results_dir / f"{instance_base}.csv"
        
        result_data = {
            'instance_name': self.instance_name,
            'dataset': self.dataset,
            'seed': self.random_seed,
            'solver_name': self.solver_name,
            'energy': computed_energy, 
            'time': time_taken,
            'local_search': self.use_local_search,
            'time_limit': self.time_limit,
            'threads': self.threads,
            'gurobi_energy': gurobi_energy,
            'gap': gap,
            'lower_bound': lower_bound,
            'status': status,
            'best_solution_time': best_solution_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame()
        
        if not df.empty:
            duplicate_mask = (
                (df['instance_name'] == result_data['instance_name']) &
                (df['dataset'] == result_data['dataset']) &
                (df['seed'] == result_data['seed']) &
                (df['solver_name'] == result_data['solver_name'])
            )
            
            if duplicate_mask.any():
                df.loc[duplicate_mask, list(result_data.keys())] = list(result_data.values())
            else:
                df = pd.concat([df, pd.DataFrame([result_data])], ignore_index=True)
        else:
            df = pd.DataFrame([result_data])
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        status_text = "OPTIMAL" if status == gp.GRB.OPTIMAL else "TIME_LIMIT" if status == gp.GRB.TIME_LIMIT else f"STATUS_{status}"
        
        print(f"Results saved to: {csv_file}")
        print(f"Energy (computed): {computed_energy:.6f}")
        print(f"Energy (Gurobi): {gurobi_energy:.6f}")
        print(f"Lower bound: {lower_bound:.6f}")
        print(f"Gap: {gap:.6f} ({gap*100:.2f}%)")
        print(f"Status: {status_text}")
        print(f"Time: {time_taken:.4f}s")
        if best_solution_time is not None:
            print(f"Best solution time: {best_solution_time:.4f}s")
        print("-" * 40)
