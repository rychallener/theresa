"""
Logging utility for comparing jaxoplanet outputs with original Theresa implementation.
"""
import numpy as np
import json
import os
from datetime import datetime

class JaxoplanetLogger:
    def __init__(self, log_file=None):
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"jaxoplanet_log_{timestamp}.txt"
        self.log_file = log_file
        self.log_dir = os.path.dirname(log_file) or "."
        os.makedirs(self.log_dir, exist_ok=True)

        # Clear the log file at initialization
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("JAXOPLANET IMPLEMENTATION LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

    def log_function_call(self, func_name, inputs=None, outputs=None, notes=None):
        """
        Log a function call with its inputs and outputs.

        Parameters
        ----------
        func_name : str
            Name of the function being called
        inputs : dict, optional
            Dictionary of input parameter names and values
        outputs : dict, optional
            Dictionary of output names and values
        notes : str, optional
            Additional notes or context
        """
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"FUNCTION: {func_name}\n")
            f.write(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write("-"*80 + "\n")

            if notes:
                f.write(f"NOTES: {notes}\n")
                f.write("-"*80 + "\n")

            if inputs:
                f.write("INPUTS:\n")
                for key, value in inputs.items():
                    f.write(f"  {key}:\n")
                    f.write(self._format_value(value, indent=4))
                f.write("-"*80 + "\n")

            if outputs:
                f.write("OUTPUTS:\n")
                for key, value in outputs.items():
                    f.write(f"  {key}:\n")
                    f.write(self._format_value(value, indent=4))

            f.write("="*80 + "\n")

    def _format_value(self, value, indent=2):
        """Format a value for logging with proper indentation."""
        prefix = " " * indent

        if value is None:
            return f"{prefix}None\n"

        # Handle numpy/jax arrays
        if hasattr(value, 'shape'):
            arr = np.array(value)
            result = f"{prefix}type: {type(value).__name__}\n"
            result += f"{prefix}shape: {arr.shape}\n"
            result += f"{prefix}dtype: {arr.dtype}\n"

            if arr.size > 0:
                result += f"{prefix}min: {np.min(arr):.10e}\n"
                result += f"{prefix}max: {np.max(arr):.10e}\n"
                result += f"{prefix}mean: {np.mean(arr):.10e}\n"
                result += f"{prefix}std: {np.std(arr):.10e}\n"

            # Show first few elements for small arrays
            if arr.size <= 10:
                result += f"{prefix}values: {arr.flatten()}\n"
            else:
                result += f"{prefix}first 5: {arr.flatten()[:5]}\n"
                result += f"{prefix}last 5: {arr.flatten()[-5:]}\n"

            return result

        # Handle scalars
        elif isinstance(value, (int, float, np.integer, np.floating)):
            return f"{prefix}{value:.10e}\n"

        # Handle strings
        elif isinstance(value, str):
            return f"{prefix}'{value}'\n"

        # Handle tuples/lists
        elif isinstance(value, (tuple, list)):
            result = f"{prefix}type: {type(value).__name__}, length: {len(value)}\n"
            if len(value) <= 5:
                for i, item in enumerate(value):
                    result += f"{prefix}[{i}]:\n"
                    result += self._format_value(item, indent=indent+2)
            else:
                result += f"{prefix}(showing first 2 and last 2 elements)\n"
                for i in [0, 1]:
                    result += f"{prefix}[{i}]:\n"
                    result += self._format_value(value[i], indent=indent+2)
                result += f"{prefix}...\n"
                for i in [-2, -1]:
                    result += f"{prefix}[{i}]:\n"
                    result += self._format_value(value[i], indent=indent+2)
            return result

        # Handle objects with attributes
        elif hasattr(value, '__dict__'):
            return f"{prefix}type: {type(value).__name__}\n{prefix}repr: {repr(value)}\n"

        # Default
        else:
            return f"{prefix}type: {type(value).__name__}, value: {value}\n"

    def log_comparison(self, func_name, jax_output, starry_output, tolerance=1e-10):
        """
        Log a comparison between jaxoplanet and starry outputs.

        Parameters
        ----------
        func_name : str
            Name of the function being compared
        jax_output : array-like
            Output from jaxoplanet implementation
        starry_output : array-like
            Output from original starry implementation
        tolerance : float
            Tolerance for considering values equal
        """
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"COMPARISON: {func_name}\n")
            f.write(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write("-"*80 + "\n")

            jax_arr = np.array(jax_output)
            starry_arr = np.array(starry_output)

            f.write(f"Jaxoplanet shape: {jax_arr.shape}\n")
            f.write(f"Starry shape: {starry_arr.shape}\n")

            if jax_arr.shape == starry_arr.shape:
                diff = np.abs(jax_arr - starry_arr)
                rel_diff = np.abs(diff / (np.abs(starry_arr) + 1e-20))

                f.write(f"\nAbsolute difference:\n")
                f.write(f"  max: {np.max(diff):.10e}\n")
                f.write(f"  mean: {np.mean(diff):.10e}\n")
                f.write(f"  std: {np.std(diff):.10e}\n")

                f.write(f"\nRelative difference:\n")
                f.write(f"  max: {np.max(rel_diff):.10e}\n")
                f.write(f"  mean: {np.mean(rel_diff):.10e}\n")

                match = np.allclose(jax_arr, starry_arr, atol=tolerance, rtol=tolerance)
                f.write(f"\nOutputs match (tolerance={tolerance}): {match}\n")

                if not match:
                    n_diff = np.sum(diff > tolerance)
                    f.write(f"Number of elements exceeding tolerance: {n_diff}/{jax_arr.size}\n")
                    f.write(f"Percentage different: {100*n_diff/jax_arr.size:.2f}%\n")
            else:
                f.write("\nWARNING: Shapes don't match! Cannot compare.\n")

            f.write("="*80 + "\n")

# Global logger instance
_logger = None

def get_logger(log_file=None):
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = JaxoplanetLogger(log_file)
    return _logger

def set_log_file(log_file):
    """Set a new log file and reset the logger."""
    global _logger
    _logger = JaxoplanetLogger(log_file)
    return _logger
