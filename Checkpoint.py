import torch

def save_model(model, file_name):
    torch.save(
        model.state_dict(),
        file_name
    )

def load_model(initialised_model, file_name):
    initialised_model.load_state_dict(
        torch.load(file_name)
    )

    initialised_model.eval()

def save_model_optimizer(model, optimizer, file_name):
    states = {
        "model_state_dict": model.state_dict(),
        "optimizer_model_dict": optimizer.state_dict()
    }

    torch.save(
        states,
        file_name
    )

def load_model_optimizer(initialised_model, initialised_optimizer, file_name):
    #loading model state
    initialised_model.load_state_dict(
        torch.load(file_name)["model_state_dict"]
    )
    initialised_model.eval()

    #loading optimizer state
    initialised_optimizer.load_state_dict(
        torch.load(file_name)["optimizer_model_dict"]
    )