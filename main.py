import torch
import sys
import os
from torch.utils.data import DataLoader
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
from data import PortfolioDataset
from losses import SharpeLoss
from layers import SineActivation
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import pandas as pd
from absl import app, flags
import argparse
import optuna
from engine import train_one_epoch, eval_one_epoch
import wandb
torch.autograd.set_detect_anomaly(True)

from pympler.tracker import SummaryTracker
tracker = SummaryTracker()


device = torch.device('cuda')
wandb.login(key="3929a6207cea3c4c0af4b77c8d0803a5f636e77c")

DEBUG = False

if not torch.cuda.is_available():
    sys.exit()

train_dataset = PortfolioDataset()
eval_dataset = PortfolioDataset(eval=True)
assert train_dataset.valid_stocks == eval_dataset.valid_stocks
args = flags.FLAGS

flags.DEFINE_integer("efd", 16, '')
flags.DEFINE_integer("dfd", 16, '')
flags.DEFINE_integer("eah", 4, '')
flags.DEFINE_integer("dah", 4, '')
flags.DEFINE_integer("el", 4, '')
flags.DEFINE_integer("dl", 4, '')
flags.DEFINE_integer("bs", 64, '')
flags.DEFINE_float("drp", 0.2, '')
flags.DEFINE_float("lr", 0.001, '')

def main(trial):
    
    # efd = trial.suggest_categorical('efd', [8, 16, 32])
    # dfd = trial.suggest_categorical('dfd', [8, 16, 32])
    # eah = trial.suggest_categorical('eah', [2, 4])
    # dah = trial.suggest_categorical('dah', [2, 4])
    # el = trial.suggest_categorical('el', [2, 4])
    # dl = trial.suggest_categorical('dl', [2, 4])
    # drp = trial.suggest_categorical('drp', [0.1, 0.2, 0.3, 0.4])
    # bs = trial.suggest_categorical('bs', [32, 64, 128])
    # lr = trial.suggest_categorical('drp', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])

    # Best model parameters as per Optuna Search
    efd = dfd = 32
    eah = 2
    dah = 2
    el = dl = 4
    drp = 0.3
    bs = 64
    lr = 1e-2
 
    config = {
        "Batch Size": bs,
        "Encoder Layer": el,
        "Decoder Layer": dl,
        "Encoder Attention Head": eah,
        "Decoder Attention Head": dah,
        "Encoder FFN Dim": efd,
        "Decoder FFN Dim": dfd,
        "Dropout": drp,
    }
    time2vec = SineActivation(3, 32)
    wandb.init(project=f"BarterAI", config=config, name=f"dropout_{drp}_bs_{bs}_heads_{eah}_layer_{el}_dim_{efd}_constantlr_fixed_randomness")
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=16, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False)

    configuration = TimeSeriesTransformerConfig(
        input_size=train_dataset.num_stocks,
        prediction_length=train_dataset.pred_len,
        context_length=train_dataset.context_len,
        lags_sequence=[0], ### FIX THIS
        num_dynamic_real_features=0,
        num_static_categorical_features=0,
        num_static_real_features=0,
        num_time_features=3,
        scaling=False,
        # cardinality=[],
        # embedding_dimension=[1],
        encoder_ffn_dim=efd,
        decoder_ffn_dim=dfd,
        encoder_attention_heads=eah,
        decoder_attention_heads=dah,
        encoder_layers=el,
        decoder_layers=dl,
        dropout=drp
    )
    model = TimeSeriesTransformerModel(configuration)
    
    constrain_head = torch.nn.Sequential(
        torch.nn.Linear(model.decoder.layers[-1].fc2.out_features, train_dataset.num_stocks),
        # torch.nn.Linear(126, train_dataset.num_stocks),
        # torch.nn.AvgPool1d(68),
        torch.nn.Softmax(dim=2)
    )
    # input = torch.ones(16, 300, 92)
    # f = model(past_values=input, past_time_features = torch.ones(16, 300, 3), past_observed_mask = torch.ones(16, 300))

    # sys.exit()
    optimizer = AdamW(list(model.parameters()) + list(time2vec.parameters()) + list(constrain_head.parameters()), lr=lr)
    # type: ignore
    loss_fn = SharpeLoss()

    num_epochs = 600
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=2*len(train_dataset),
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    model.to(device)
    constrain_head.to(device)
    # model.load_state_dict(torch.load("./model.pth"))
    # wandb.save("./main.py") 
    model.eval()
    best_sharpe_train = 0
    best_sharpe_eval = 0
    for epoch in range(num_epochs):
        sharpe_train, weights_train = train_one_epoch(model, constrain_head, optimizer, lr_scheduler, loss_fn, train_dataloader, progress_bar, time2vec)
        sharpe_eval, weights_eval = train_one_epoch(model, constrain_head, optimizer, lr_scheduler, loss_fn, eval_dataloader, progress_bar, time2vec, eval=True)
        # sharpe_eval = 
        # sharpe_eval = eval_one_epoch(model, eval_dataloader, time2vec, constrain_head, loss_fn)
        if sharpe_train > best_sharpe_train:
            torch.save(weights_train, "weights_best_train")
            best_sharpe_train = sharpe_train
        if sharpe_eval > best_sharpe_eval:
            print("Saving model with Sharpe: ", sharpe_eval)
            torch.save(model.state_dict(), 'model_.pth')
            best_sharpe = sharpe_eval
            torch.save(weights_eval, "weights_best_eval")
            torch.save(weights_train, "weights_best_train_as_per_eval")
            # print(weights_eval.sort())
            wandb.log({"Epoch": epoch})
        trial.report(best_sharpe, epoch)
    wandb.save("./main.py")  
    wandb.save("./engine.py")                     
    wandb.save("./data.py")
    return best_sharpe


if __name__ == '__main__':
    # app.run(main) 
    study = optuna.create_study(storage="sqlite:///db.sqlite3",
                                direction="maximize")
    study.optimize(main, n_trials=50)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
   
     