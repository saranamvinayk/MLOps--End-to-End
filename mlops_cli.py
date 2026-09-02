#!/usr/bin/env python3
import os
import sys
import argparse

class Style:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

MODELS = {
    "1": {"name": "Random Forest Regressor", "script": "python train.py --model rf"},
    "2": {"name": "Gradient Boosting Regressor", "script": "python train.py --model gbm"},
    "3": {"name": "Ridge Linear Regression", "script": "python train.py --model lr"}
}

DEPLOYMENTS = {
    "1": {"name": "Local Docker (FastAPI) & Monitoring", "cmd": "docker-compose up --build -d"},
    "2": {"name": "BentoML Enterprise Serving (Local)", "cmd": "bentoml serve service_bento:HousePriceService"},
    "3": {"name": "Serverless (Google Cloud Run)", "cmd": "gcloud run services replace cloudrun-service.yaml"},
    "4": {"name": "Kubernetes Cluster", "cmd": "kubectl apply -f k8s/deployment.yaml"},
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
    parser = argparse.ArgumentParser(description="MLOps Pipeline Controller")
    parser.add_argument("--model", type=str, choices=["rf", "gbm", "lr"], help="Bypass UI to select model")
    parser.add_argument("--deploy", type=str, choices=["docker", "bentoml", "serverless", "k8s"], help="Bypass UI to select deployment")
    args = parser.parse_args()

    print(f"{Style.BOLD}{Style.GREEN}🚀 MLOps Pipeline Orchestrator v2{Style.END}")

    # 1. Model Selection
    selected_model = None
    if args.model:
        flag_to_key = {"rf": "1", "gbm": "2", "lr": "3"}
        selected_model = MODELS[flag_to_key[args.model]]
    else:
        selected_model = interactive_prompt("Select Training Model", MODELS)

    execute_step(f"Training {selected_model['name']}", selected_model['script'])
    
    # 2. Deployment Selection
    selected_deploy = None
    if args.deploy:
        flag_to_key = {"docker": "1", "bentoml": "2", "serverless": "3", "k8s": "4"}
        selected_deploy = DEPLOYMENTS[flag_to_key[args.deploy]]
    else:
        selected_deploy = interactive_prompt("Select Deployment Target", DEPLOYMENTS)

    execute_step(f"Deploying to {selected_deploy['name']}", selected_deploy['cmd'])

if __name__ == "__main__":
    main()
