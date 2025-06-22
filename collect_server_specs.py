#!/usr/bin/env python3
"""
Collect server specifications for the experiment report
"""

import subprocess
import platform
import psutil
import duckdb
from datetime import datetime

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def collect_server_specs():
    """Collect comprehensive server specifications"""
    
    specs = {
        "report_generated": datetime.now().isoformat(),
        "hardware": {},
        "operating_system": {},
        "python_environment": {},
        "duckdb_configuration": {}
    }
    
    specs["hardware"]["cpu_model"] = run_command("lscpu | grep 'Model name' | cut -d':' -f2 | xargs")
    specs["hardware"]["cpu_cores"] = psutil.cpu_count(logical=False)
    specs["hardware"]["cpu_threads"] = psutil.cpu_count(logical=True)
    specs["hardware"]["cpu_architecture"] = platform.machine()
    
    memory = psutil.virtual_memory()
    specs["hardware"]["total_memory_gb"] = round(memory.total / (1024**3), 2)
    specs["hardware"]["available_memory_gb"] = round(memory.available / (1024**3), 2)
    specs["hardware"]["memory_usage_percent"] = memory.percent
    
    specs["operating_system"]["os_name"] = platform.system()
    specs["operating_system"]["os_release"] = platform.release()
    specs["operating_system"]["os_version"] = platform.version()
    specs["operating_system"]["distribution"] = run_command("lsb_release -d | cut -d':' -f2 | xargs")
    specs["operating_system"]["kernel"] = platform.release()
    
    specs["python_environment"]["python_version"] = platform.python_version()
    specs["python_environment"]["python_implementation"] = platform.python_implementation()
    specs["python_environment"]["uv_version"] = run_command("uv --version")
    
    specs["duckdb_configuration"]["duckdb_version"] = duckdb.__version__
    
    try:
        conn = duckdb.connect()
        vss_result = conn.execute("SELECT * FROM duckdb_extensions() WHERE extension_name = 'vss'").fetchall()
        specs["duckdb_configuration"]["vss_extension"] = "Available" if vss_result else "Not Available"
        if vss_result:
            specs["duckdb_configuration"]["vss_extension_details"] = vss_result[0]
        conn.close()
    except Exception as e:
        specs["duckdb_configuration"]["vss_extension"] = f"Error checking: {e}"
    
    return specs

def format_specs_report(specs):
    """Format specifications as a readable report"""
    
    report = []
    report.append("🖥️ Server Specifications Report")
    report.append("=" * 50)
    report.append("")
    report.append(f"📅 Report Generated: {specs['report_generated']}")
    report.append("")
    
    report.append("🔧 Hardware Specifications:")
    report.append(f"   CPU Model: {specs['hardware']['cpu_model']}")
    report.append(f"   CPU Cores: {specs['hardware']['cpu_cores']} physical cores")
    report.append(f"   CPU Threads: {specs['hardware']['cpu_threads']} logical threads")
    report.append(f"   CPU Architecture: {specs['hardware']['cpu_architecture']}")
    report.append(f"   Total Memory: {specs['hardware']['total_memory_gb']} GB")
    report.append(f"   Available Memory: {specs['hardware']['available_memory_gb']} GB")
    report.append(f"   Memory Usage: {specs['hardware']['memory_usage_percent']:.1f}%")
    report.append("")
    
    report.append("💿 Operating System:")
    report.append(f"   OS: {specs['operating_system']['distribution']}")
    report.append(f"   Kernel: {specs['operating_system']['kernel']}")
    report.append(f"   Platform: {specs['operating_system']['os_name']}")
    report.append(f"   Architecture: {specs['hardware']['cpu_architecture']}")
    report.append("")
    
    report.append("🐍 Python Environment:")
    report.append(f"   Python Version: {specs['python_environment']['python_version']}")
    report.append(f"   Python Implementation: {specs['python_environment']['python_implementation']}")
    report.append(f"   UV Version: {specs['python_environment']['uv_version']}")
    report.append("")
    
    report.append("🗄️ DuckDB Configuration:")
    report.append(f"   DuckDB Version: {specs['duckdb_configuration']['duckdb_version']}")
    report.append(f"   VSS Extension: {specs['duckdb_configuration']['vss_extension']}")
    report.append("")
    
    return "\n".join(report)

if __name__ == "__main__":
    print("🔍 Collecting server specifications...")
    specs = collect_server_specs()
    
    import json
    with open("server_specifications.json", "w") as f:
        json.dump(specs, f, indent=2)
    
    report = format_specs_report(specs)
    with open("server_specifications.txt", "w") as f:
        f.write(report)
    
    print("✅ Server specifications collected:")
    print("   - server_specifications.json (structured data)")
    print("   - server_specifications.txt (readable report)")
    print("")
    print(report)
