import traceback
import sys

try:
    with open("benchmark_quality_output.txt", "w") as f:
        sys.stdout = f
        sys.stderr = f
        
        # Load and run the script
        with open("benchmark_quality.py", "r", encoding="utf-8") as script_file:
            script_code = script_file.read()
            
        exec(script_code, globals())
except Exception as e:
    with open("benchmark_quality_output.txt", "a") as f:
        f.write("\n\n=== EXCEPTION ===\n")
        traceback.print_exc(file=f)
