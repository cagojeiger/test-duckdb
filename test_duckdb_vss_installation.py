#!/usr/bin/env python3
"""
Test DuckDB VSS extension installation
"""

import duckdb


def test_vss_installation() -> bool:
    """Test DuckDB VSS extension installation"""
    print("🔧 Testing DuckDB VSS extension installation...")

    try:
        conn = duckdb.connect()

        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")

        result = conn.execute(
            "SELECT * FROM duckdb_extensions() WHERE extension_name = 'vss'"
        ).fetchall()

        if result:
            print("✅ DuckDB VSS extension installed and loaded successfully")
            print(f"   Extension info: {result[0]}")
            return True
        else:
            print("❌ VSS extension not found in loaded extensions")
            return False

    except Exception as e:
        print(f"❌ Failed to install/load VSS extension: {e}")
        return False
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    success = test_vss_installation()
    exit(0 if success else 1)
