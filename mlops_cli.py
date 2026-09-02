#!/usr/bin/env python3
import os
import sys
import time
import argparse

class Style:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

MODELS = {
    "1": {"name": "Random Forest", "script": "python train.py"},
}

DEPLOYMENTS = {
    "1": {"name": "Local Docker (FastAPI)", "cmd": "docker-compose up --build -d"},
    "2": {"name": "Kubernetes Cluster", "cmd": "kubectl apply -f k8s/deployment.yaml"},
}

def interactive_prompt(title, options):
    print(f"\n{Style.BOLD}{Style.BLUE}=== {title} ==={Style.END}")
    for key, value in options.items():
        print(f"  [{Style.GREEN}{key}{Style.END}] {value['name']}")
    print(f"  [{Style.RED}q{Style.END}] Quit\n")
    
    while True:
        choice = input(f"{Style.BOLD}Select an option > {Style.END}").strip().lower()
        if choice == 'q': sys.exit(0)
        if choice in options: return options[choice]
        print(f"{Style.RED}Invalid choice.{Style.END}")

def execute_step(step_name, command):
    print(f"\n{Style.YELLOW}Starting:{Style.END} {step_name}")
    print(f"{Style.BLUE}Executing: {command}{Style.END}\n")
    try:
        os.system(command)
        print(f"\n{Style.GREEN}✔ {step_name} completed!{Style.END}")
    except KeyboardInterrupt:
        sys.exit(1)

def main():
    print(f"{Style.BOLD}{Style.GREEN}🚀 MLOps Pipeline Orchestrator{Style.END}")
    selected_model = interactive_prompt("Select Training Model", MODELS)
    execute_step(f"Training {selected_model['name']}", selected_model['script'])
    
    selected_deploy = interactive_prompt("Select Deployment Target", DEPLOYMENTS)
    execute_step(f"Deploying to {selected_deploy['name']}", selected_deploy['cmd'])

if __name__ == "__main__":
    main()
