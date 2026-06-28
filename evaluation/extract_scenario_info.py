import os
from pathlib import Path
from collections import defaultdict

def extract_agents_from_scenarios(root_dir):
    """
    Extract agent IDs for each scenario in the OPV2V dataset.
    
    Args:
        root_dir: Root directory of opv2v_data_dumping
    
    Returns:
        Dictionary with structure: {split: {scenario: [agent_ids]}}
    """
    root_path = Path(r"opv2v_data_dumping/")
    results = defaultdict(dict)
    
    # Iterate through each split (train, validate, test, test_culvercity)
    for split_dir in root_path.iterdir():
        if not split_dir.is_dir():
            continue
            
        split_name = split_dir.name
        print(f"\n{'='*60}")
        print(f"Processing split: {split_name}")
        print(f"{'='*60}")
        
        # Iterate through each scenario
        for scenario_dir in split_dir.iterdir():
            if not scenario_dir.is_dir():
                continue
                
            scenario_name = scenario_dir.name
            agent_ids = []
            
            # Find all agent directories (numeric folders)
            for item in scenario_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    agent_ids.append(item.name)
            
            # Sort agent IDs for consistent output
            agent_ids.sort()
            
            # Store results
            results[split_name][scenario_name] = agent_ids
            
            # Print results for this scenario
            print(f"\nScenario: {scenario_name}")
            print(f"  Number of agents: {len(agent_ids)}")
            print(f"  Agent IDs: {', '.join(agent_ids)}")
    
    return results

def save_results_to_file(results, output_file="agent_summary.txt"):
    """
    Save the extraction results to a text file.
    
    Args:
        results: Dictionary containing the extraction results
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("OPV2V Dataset - Agent Summary\n")
        f.write("="*60 + "\n\n")
        
        for split_name, scenarios in results.items():
            f.write(f"\nSplit: {split_name}\n")
            f.write("-"*60 + "\n")
            
            total_agents = 0
            for scenario_name, agent_ids in scenarios.items():
                f.write(f"\nScenario: {scenario_name}\n")
                f.write(f"  Number of agents: {len(agent_ids)}\n")
                f.write(f"  Agent IDs: {', '.join(agent_ids)}\n")
                total_agents += len(agent_ids)
            
            f.write(f"\nTotal scenarios in {split_name}: {len(scenarios)}\n")
            f.write(f"Total agents in {split_name}: {total_agents}\n")
            f.write("\n" + "="*60 + "\n")

def generate_csv_report(results, output_file="agent_report.csv"):
    """
    Generate a CSV report of agents per scenario.
    
    Args:
        results: Dictionary containing the extraction results
        output_file: Output CSV file path
    """
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Split', 'Scenario', 'Number of Agents', 'Agent IDs'])
        
        for split_name, scenarios in results.items():
            for scenario_name, agent_ids in scenarios.items():
                writer.writerow([
                    split_name,
                    scenario_name,
                    len(agent_ids),
                    ', '.join(agent_ids)
                ])

# Main execution
if __name__ == "__main__":
    # Set the root directory path
    root_directory = "opv2v_data_dumping"
    
    # Extract agent information
    results = extract_agents_from_scenarios(root_directory)
    
    # Save results to text file
    save_results_to_file(results, "agent_summary.txt")
    print(f"\n\nResults saved to: agent_summary.txt")
    
    # Generate CSV report
    generate_csv_report(results, "agent_report.csv")
    print(f"CSV report saved to: agent_report.csv")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for split_name, scenarios in results.items():
        total_agents = sum(len(agents) for agents in scenarios.values())
        print(f"{split_name}: {len(scenarios)} scenarios, {total_agents} total agents")
