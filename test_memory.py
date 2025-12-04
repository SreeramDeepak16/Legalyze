import os
import sys

# Add current directory to sys.path
sys.path.append(os.getcwd())

from episodic_memory_agent import memory, format_event

def test_memory():
    print("Testing Episodic Memory Manager...")
    
    # Test Adding Memory
    print("\n1. Testing Add Memory...")
    event_id = memory.add_or_update(
        event_type="meeting",
        actor="Alice",
        summary="Met with Alice to discuss project",
        details="Discussed the new feature roadmap and timelines."
    )
    print(f"Memory added with ID: {event_id}")
    
    # Test Retrieval
    print("\n2. Testing Retrieve Memory...")
    results = memory.search("Alice project")
    print(f"Found {len(results)} results.")
    for res in results:
        print(format_event(res))
        
    # Test List Recent
    print("\n3. Testing List Recent...")
    recent = memory.list_recent(days=1)
    print(f"Found {len(recent)} recent events.")
    
    print("\nTest Complete.")

if __name__ == "__main__":
    test_memory()
