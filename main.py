import yaml

from train_system import TrainSystem

if __name__=="__main__":
    with open("./hparams.yaml", "r") as file:
        config = yaml.safe_load(file)


    system = TrainSystem(config)
    #system.eval_iteration(0, load_checkpoint="./vit-t-mae.pt")
    system.train()