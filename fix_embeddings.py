#!/usr/bin/env python3
"""
Simple script to check and fix embeddings in the Supabase database.
"""
import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get Supabase connection info
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    sys.exit(1)

print(f"Connecting to Supabase at {SUPABASE_URL}")

# Helper function to make Supabase REST API calls
def supabase_request(method, endpoint, data=None, params=None):
    """Make a request to the Supabase REST API."""
    url = f"{SUPABASE_URL}/{endpoint}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    if method == "GET":
        response = requests.get(url, headers=headers, params=params)
    elif method == "POST":
        response = requests.post(url, headers=headers, json=data)
    elif method == "DELETE":
        response = requests.delete(url, headers=headers, params=params)
    
    if response.status_code >= 400:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
    return response.json()

def check_chunk_embeddings():
    """Check if chunk embeddings look valid."""
    print("Checking chunk embeddings...")
    
    # Get a sample of chunks
    chunks = supabase_request("GET", "rest/v1/chunks", params={"select": "id,embedding", "limit": 10})
    
    if not chunks or len(chunks) == 0:
        print("No chunks found in the database")
        return
    
    print(f"Found {len(chunks)} chunks to examine")
    
    # Check for issues
    suspicious_chunks = []
    for chunk in chunks:
        chunk_id = chunk.get("id")
        embedding = chunk.get("embedding", [])
        
        # Check if embedding is not a list or empty
        if not embedding or not isinstance(embedding, list):
            print(f"Chunk {chunk_id}: Invalid embedding format: {type(embedding)}")
            suspicious_chunks.append((chunk_id, "invalid format"))
            continue
        
        # Check if embedding is too short
        if len(embedding) < 1536:
            print(f"Chunk {chunk_id}: Embedding too short ({len(embedding)} values)")
            suspicious_chunks.append((chunk_id, "too short"))
            continue
        
        # Check if first few values look suspicious (all the same)
        first_values = embedding[:5]
        if all(abs(v - first_values[0]) < 0.0001 for v in first_values[1:]):
            print(f"Chunk {chunk_id}: Suspicious embedding pattern - all similar values: {first_values}")
            suspicious_chunks.append((chunk_id, "suspicious pattern"))
            continue
        
        print(f"Chunk {chunk_id}: Embedding looks valid. First 5 values: {first_values}")
    
    return suspicious_chunks

def delete_chunks(chunk_ids):
    """Delete chunks with the specified IDs."""
    print(f"Deleting {len(chunk_ids)} chunks with suspicious embeddings...")
    
    for chunk_id in chunk_ids:
        response = supabase_request("DELETE", "rest/v1/chunks", params={"id": f"eq.{chunk_id}"})
        if response is not None:
            print(f"Successfully deleted chunk {chunk_id}")
        else:
            print(f"Failed to delete chunk {chunk_id}")

def main():
    """Main function to check and fix embeddings."""
    print("Starting embedding diagnostic...")
    
    # Check chunk embeddings
    suspicious_chunks = check_chunk_embeddings()
    
    if not suspicious_chunks:
        print("No suspicious embeddings found. Your embeddings look valid!")
        return
    
    print(f"Found {len(suspicious_chunks)} suspicious chunks")
    
    # Confirm deletion
    confirm = input(f"Do you want to delete these {len(suspicious_chunks)} suspicious chunks? (y/n): ")
    if confirm.lower() == 'y':
        chunk_ids = [chunk_id for chunk_id, _ in suspicious_chunks]
        delete_chunks(chunk_ids)
        print("Cleanup completed. You should now re-ingest your documents with proper embeddings.")
    else:
        print("No changes made. Review the embedding issues and fix them manually.")

if __name__ == "__main__":
    main() 