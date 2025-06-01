#!/usr/bin/env python3
"""
Script to initialize the Supabase database with the required tables and extensions.
"""
import logging
import os
import sys

# Add the parent directory to the path so we can import our package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.database import supabase_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def initialize_database():
    """
    Initialize the Supabase database with the required tables and extensions.
    """
    # Get the setup SQL from the setup.sql file
    setup_sql_path = os.path.join(os.path.dirname(__file__), 'setup.sql')
    
    try:
        with open(setup_sql_path, 'r') as f:
            setup_sql = f.read()
    except Exception as e:
        logger.error(f"Failed to read setup SQL: {e}")
        return False
    
    # Execute the setup SQL
    try:
        client = supabase_manager.client
        logger.info("Successfully connected to Supabase")
        
        # Split the SQL into individual statements
        statements = setup_sql.split(';')
        
        for statement in statements:
            # Skip empty statements
            if not statement.strip():
                continue
                
            # Execute the statement
            logger.info(f"Executing SQL: {statement}")
            result = client.table('_').query(statement + ';').execute()
            logger.info(f"Result: {result}")
        
        logger.info("Database initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

if __name__ == "__main__":
    initialize_database() 