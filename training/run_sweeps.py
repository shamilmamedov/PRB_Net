import wandb
import yaml
import fire

from training import run_experiment

def main(config: str):
    with open(config, 'r') as file:
        sweep_config = yaml.safe_load(file)
    
    sweep_id = wandb.sweep(sweep=sweep_config, project='learning-dlo-dynamics-with-rnn')
    wandb.agent(sweep_id=sweep_id, function=run_experiment.main, count=10)


if __name__ == '__main__': 
    fire.Fire(main)
