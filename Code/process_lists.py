# Function to strip external quotes and convert to list 
import ast


def process_entry(entry):
    try:
        # Remove external quotes and convert to list
        entry = entry.strip('"')
        return ast.literal_eval(entry)
    except (ValueError, SyntaxError):
        # Handle malformed entries
        #print(f"Failed to process: {entry}")
        return None