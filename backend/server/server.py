import flwr as fl

def main():
    # Define strategy utilizing Federated Averaging
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        # Mandatory configured bounds: 3 clients minimal 
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    print("Deploying Federated Learning Server.")
    print("Strategy [FedAvg] Bound. Waiting for 3 minimum clients to begin.")

    # Start built in Flower server instance mapping to localhost
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
